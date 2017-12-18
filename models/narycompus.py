"""
Define the model architecture and define forward and predict computations.
"""
from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional

import model_utils as mu


class BPRLoss(torch.nn.Module):
    """
    The Bayesian Personalized Ranking loss.

     References
    ----------
    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """
    def __init__(self, size_average=False):
        super(BPRLoss, self).__init__()
        self.size_average = size_average

    def forward(self, true_ex_scores, false_ex_scores):
        """
        Push true scores apart from false scores.
        :param true_ex_scores: torch Variable; (batch_size, 1)
        :param false_ex_scores: torch Variable; (batch_size, 1)
        :return: torch Variable.
        """
        diff = true_ex_scores - false_ex_scores
        caseloss = -1.0 * functional.logsigmoid(diff)
        if self.size_average:
            loss = caseloss.mean()
        else:
            loss = caseloss.sum()
        return loss


class NaryCompUSchema(torch.nn.Module):
    """
    Use LSTMs to compose surface text relations and entities to score the
    sentence as being a possible n-ary relation.
    """
    def __init__(self, word2idx, embedding_path, num_layers=1,
                 embedding_dim=100, hidden_dim=50, dropout=0.3):
        super(NaryCompUSchema, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        # Voacb info.
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in self.word2idx.iteritems()}
        self.vocab_size = len(self.word2idx)

        # Define the elements of the architecture.
        self.in_drop = torch.nn.Dropout(p=dropout)
        self.embeddings = mu.init_pretrained_embed(
            embed_path=embedding_path, word2idx=word2idx,
            embedding_dim=embedding_dim)
        # The column lstm to compose surface forms.
        self.col_lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim)
        # The row lstm to compose entities.
        self.row_lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim)
        # Dropout out at both the lstm hidden outputs.
        self.colh_drop = torch.nn.Dropout(p=dropout)
        self.rowh_drop = torch.nn.Dropout(p=dropout)
        self.criterion_bpr = BPRLoss(size_average=False)

        # Move model to the GPU.
        if torch.cuda.is_available():
            print('Running on GPU.')
            self.in_drop = self.in_drop.cuda()
            self.embeddings = self.embeddings.cuda()
            self.col_lstm = self.col_lstm.cuda()
            self.row_lstm = self.row_lstm.cuda()
            self.colh_drop = self.colh_drop.cuda()
            self.rowh_drop = self.rowh_drop.cuda()
            self.criterion_bpr = self.criterion_bpr.cuda()

    def forward(self, batch_cr, batch_neg):
        """
        Pass through a forward pass and return the loss.
        :param batch_cr: (batch_col_row) dict of the form:
            {'col': Torch Tensor; the padded and sorted-by-length sentence.
             'row': Torch Tensor; the padded and sorted-by-length entities.
             'col_lens': list(int); lengths of all sequences in 'col'.
             'row_lens': list(int); lengths of all sequences in 'row'.
             'sorted_colrefs': list(int); ints saying which seq in col came
                    from which document. ints in range [0, len(docs)]
             'sorted_rowrefs': list(int); ints saying which seq in row came
                    from which document. ints in range [0, len(docs)]}
        :param batch_neg: (batch_row_neg) dict of the form:
            {'row': Torch Tensor; the padded and sorted-by-length entities.
             'row_lens': list(int); lengths of all sequences in 'col'.
             'sorted_rowrefs': list(int); ints saying which seq in row came
                    from which document. ints in range [0, len(docs)]}
        :return: loss; torch Variable.
        """
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'],\
            batch_cr['row_lens'], batch_cr['sorted_colrefs'],\
            batch_cr['sorted_rowrefs']
        row_neg, row_neg_lens, row_neg_refs = \
            batch_neg['row'], batch_neg['row_lens'], batch_neg['sorted_rowrefs']

        # Pass the col and row through the appropriate lstms.
        col_hidden = self._col_compose(col=col, col_refs=col_refs,
                                       col_lengths=col_lens)
        row_hidden = self._row_compose(row=row, row_refs=row_refs,
                                       row_lengths=row_lens)
        row_neg_hidden = self._row_compose(row=row_neg, row_refs=row_neg_refs,
                                           row_lengths=row_neg_lens)
        # At this point the stuff in the hidden vectors is assumed to be
        # aligned. The compatability between the rows and the columns of
        # positive examples:
        comp_score_pos = torch.sum(col_hidden * row_hidden, dim=1)
        comp_score_neg = torch.sum(col_hidden * row_neg_hidden, dim=1)
        loss = self.criterion_bpr(true_ex_scores=comp_score_pos,
                                  false_ex_scores=comp_score_neg)
        return loss

    def _row_compose(self, row, row_refs, row_lengths, inference=False):
        """
        Pass through the row lstm and return its representation for a batch of
        rows.
        :param row: Torch Tensor; the padded and sorted-by-length entities.
        :param row_refs: list(int); ints saying which seq in row came
                    from which document. ints in range [0, len(docs)]}
        :param row_lengths: list(int); lengths of all sequences in 'row'.
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: agg_hidden; Torch Variable with the hidden representations for the batch.
        """
        total_sents = row.size(0)  # Batch size.
        # Make initialized hidden and cell states for lstm.
        row_h0 = torch.zeros(self.num_layers, total_sents, self.hidden_dim)
        row_c0 = torch.zeros(self.num_layers, total_sents, self.hidden_dim)
        # Make the doc masks; there must be an easier way to do this. :/
        row_refs = np.array(row_refs)
        row_masks = np.zeros((total_sents, total_sents, self.hidden_dim))
        for ref in xrange(total_sents):
            row_masks[ref, row_refs == ref, :] = 1.0
        row_masks = torch.FloatTensor(row_masks)
        # Make all model variables to Variables and move to the GPU.
        row_h0, row_c0 = Variable(row_h0, volatile=inference), \
                         Variable(row_c0, volatile=inference)
        row, row_masks = Variable(row, volatile=inference), \
                         Variable(row_masks, volatile=inference)
        if torch.cuda.is_available():
            row = row.cuda()
            row_h0, row_c0 = row_h0.cuda(), row_c0.cuda()
            row_masks = row_masks.cuda()
        # Pass forward.
        embeds = self.embeddings(row)
        if inference == False:
            embeds = self.in_drop(embeds)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, row_lengths,
                                                         batch_first=True)
        out, (hidden, cell) = self.row_lstm(packed, (row_h0, row_c0))
        # Put the hidden vectors into the unsorted order >_<.
        agg_hidden = torch.sum(hidden * row_masks, dim=1)
        if inference == False:
            agg_hidden = self.rowh_drop(agg_hidden)
        return agg_hidden

    def _col_compose(self, col, col_refs, col_lengths, inference=False):
        """
        Pass through the col lstm and return its representation for a batch of
        col. This allows me to potentially use different compositions functions
        for the row and the column.
        :param col: Torch Tensor; the padded and sorted-by-length entities.
        :param col_refs: list(int); ints saying which seq in col came
                    from which document. ints in range [0, len(docs)]}
        :param col_lengths: list(int); lengths of all sequences in 'col'.
        :param inference: boolean; if True do pure inference; turn off dropout and dont store gradients.
        :return: agg_hidden; Torch Variable with the hidden representations for the batch.
        """
        total_sents = col.size(0)  # Batch size.
        # Make initialized hidden and cell states for lstm.
        col_h0 = torch.zeros(self.num_layers, total_sents, self.hidden_dim)
        col_c0 = torch.zeros(self.num_layers, total_sents, self.hidden_dim)
        # Make the doc masks; there must be an easier way to do this. :/
        col_refs = np.array(col_refs)
        col_masks = np.zeros((total_sents, total_sents, self.hidden_dim))
        for ref in xrange(total_sents):
            col_masks[ref, col_refs == ref, :] = 1.0
        col_masks = torch.FloatTensor(col_masks)
        # Make all model variables to Variables and move to the GPU.
        col_h0, col_c0 = Variable(col_h0, volatile=inference),\
                         Variable(col_c0, volatile=inference)
        col, col_masks = Variable(col, volatile=inference), \
                         Variable(col_masks, volatile=inference)
        if torch.cuda.is_available():
            col = col.cuda()
            col_h0, col_c0 = col_h0.cuda(), col_c0.cuda()
            col_masks = col_masks.cuda()
        # Pass forward.
        embeds = self.embeddings(col)
        if inference == False:
            embeds = self.in_drop(embeds)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, col_lengths,
                                                         batch_first=True)
        out, (hidden, cell) = self.col_lstm(packed, (col_h0, col_c0))
        # Put the hidden vectors into the unsorted order; all except one
        # vec will be zeroed out for each example in the batch.
        agg_hidden = torch.sum(hidden * col_masks, dim=1)
        if inference == False:
            agg_hidden = self.colh_drop(agg_hidden)
        return agg_hidden

    def predict(self, batch_cr):
        """
        Pass through a forward pass and return the loss.
        :param batch_cr: (batch_col_row) dict of the form:
            {'col': Torch Tensor; the padded and sorted-by-length sentence.
             'row': Torch Tensor; the padded and sorted-by-length entities.
             'col_lens': list(int); lengths of all sequences in 'col'.
             'row_lens': list(int); lengths of all sequences in 'row'.
             'sorted_colrefs': list(int); ints saying which seq in col came
                    from which document. ints in range [0, len(docs)]
             'sorted_rowrefs': list(int); ints saying which seq in row came
                    from which document. ints in range [0, len(docs)]}
        :param: return_vectors; boolean, says if you want the learnt vectors.
        :return: loss; torch Variable.
        """
        col, row, col_lens, row_lens, col_refs, row_refs = \
            batch_cr['col'], batch_cr['row'], batch_cr['col_lens'],\
            batch_cr['row_lens'], batch_cr['sorted_colrefs'],\
            batch_cr['sorted_rowrefs']

        total_sents = col.size(0)
        # Pass the col and row through the appropriate lstms.
        col_hidden = self._col_compose(col=col, col_refs=col_refs,
                                       col_lengths=col_lens, inference=True)
        row_hidden = self._row_compose(row=row, row_refs=row_refs,
                                       row_lengths=row_lens, inference=True)
        # At this point the stuff in the hidden vectors is assumed to be
        # aligned. The compatability between the rows and the columns; point
        # mul the rows and sum.:
        comp_score = torch.sum(col_hidden * row_hidden, dim=1)
        probs = torch.exp(functional.logsigmoid(comp_score))

        # Make numpy arrays and return.
        if torch.cuda.is_available():
            probs = probs.cpu().data.numpy()
            col_hidden = col_hidden.cpu().data.numpy()
            row_hidden = row_hidden.cpu().data.numpy()
        else:
            probs = probs.data.numpy()
            col_hidden = col_hidden.data.numpy()
            row_hidden = row_hidden.data.numpy()

        assert(probs.shape[0] == col_hidden.shape[0] == row_hidden.shape[0]
               == total_sents)
        return probs, col_hidden, row_hidden