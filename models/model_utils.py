"""
Utilities to feed and initialize the models.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys, random
import codecs
import json, pprint

import numpy as np
import torch
from torch.autograd import Variable

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def init_pretrained_embed(embed_path, word2idx, embedding_dim):
    """
    Initialize the store for embeddings with the pre-trained embeddings
    or the random word vectors.
    :param embed_path:
    :param word2idx:
    :param embedding_dim:
    :return:
    """
    vocab_size = len(word2idx)
    # read in the glove files
    embed_file = os.path.join(embed_path, '100d.embed.json')
    with codecs.open(embed_file, 'r', 'utf-8') as fp:
        word2glove = json.load(fp)
    print('Read embeddings: {:s}'.format(embed_file))

    # then make giant matrix with all the matching vocab words
    padding_idx = 0
    # follow Karpahty's advice and initialize really small
    pretrained = torch.randn(vocab_size, embedding_dim) * 0.01
    count = 0
    for word, idx in word2idx.iteritems():
        # reserve the padding idx as 0
        if idx == padding_idx:
            torch.FloatTensor(embedding_dim).zero_()
        # keep as random initialization
        if word not in word2glove:
            count += 1
            continue
        pretrained[idx] = torch.FloatTensor(word2glove[word])

    print('Initialized with pretrained; {:d}/{:d} words not in vocab'.
          format(count, vocab_size))
    embed = torch.nn.Embedding(vocab_size, embedding_dim)
    embed.weight = torch.nn.Parameter(pretrained)
    return embed


def batched_loss(model, batcher, batch_size, full_col, full_row):
    """
    Make predictions batch by batch. Dont do any funky shuffling shit.
    :param model: the model object with a predict method.
    :param batcher: reference to model_utils.Batcher class.
    :param batch_size: int; number of docs to consider in a batch.
    :param full_col: raw data read from disk; columns
    :param full_row: raw data read from disk; rows.
    :return: preds: numpy array; predictions on int_mapped_X.
    """
    # Intialize batcher but dont shuffle.
    loss_batcher = batcher(full_col=full_col, full_row=full_row,
                           batch_size=batch_size, train_mode=True)
    loss = Variable(torch.FloatTensor([0]))
    # Hack to make it work, need to do better at moving to and from gpu.
    if torch.cuda.is_available():
        loss = loss.cuda()
    for batch_cr, batch_neg in loss_batcher.next_batch():
        loss += model.forward(batch_cr=batch_cr, batch_neg=batch_neg)
    loss = float(loss.data)
    return loss


def batched_predict(model, batcher, batch_size, full_col, full_row):
    """
    Make predictions batch by batch. Dont do any funky shuffling shit.
    :param model: the model object with a predict method.
    :param batcher: reference to model_utils.Batcher class.
    :param batch_size: int; number of docs to consider in a batch.
    :param full_col: raw data read from disk; columns
    :param full_row: raw data read from disk; rows.
    :return: preds: numpy array; predictions on int_mapped_X.
    """
    # Intialize batcher but dont shuffle.
    predict_batcher = batcher(full_col=full_col, full_row=full_row,
                              batch_size=batch_size, train_mode=False)
    probs, col_hidden, row_hidden = [], [], []
    for batch_cr in predict_batcher.next_batch():
        batch_probs, batch_col_hidden, batch_row_hidden = \
            model.predict(batch_cr=batch_cr)
        probs.append(batch_probs)
        col_hidden.append(batch_col_hidden)
        row_hidden.append(batch_row_hidden)
    probs = np.hstack(probs)
    col_hidden = np.vstack(col_hidden)
    row_hidden = np.vstack(row_hidden)
    return probs, col_hidden, row_hidden


def pad_sort_seq(int_mapped_seq):
    doc_ref = range(len(int_mapped_seq))
    # Get sorted indices.
    sorted_indices = sorted(range(len(int_mapped_seq)),
                            key=lambda k: -len(int_mapped_seq[k]))
    max_length = len(int_mapped_seq[sorted_indices[0]])

    # Make the padded sequence.
    X_exp_padded = torch.LongTensor(len(int_mapped_seq), max_length).zero_()
    # Make the sentences into tensors sorted by length and place then into the
    # padded tensor.
    sorted_ref = []
    sorted_lengths = []
    for i, sent_i in enumerate(sorted_indices):
        tt = torch.LongTensor(int_mapped_seq[sent_i])
        lenght = tt.size(0)
        X_exp_padded[i, 0:lenght] = tt
        # Rearrange the doc refs.
        sorted_ref.append(doc_ref[sent_i])
        # Make this because packedpadded seq asks for it.
        sorted_lengths.append(lenght)

    return X_exp_padded, sorted_lengths, sorted_ref


def pad_sort_data(im_col_raw, im_row_raw, im_row_raw_neg=None):
    """
    Pad the data and sort such that the sentences are sorted in descending order
    of sentence length. Jumble all the sentences in this sorting but also
    maintain a list which says which sentence came from which document of the
    same length as the total number of sentences with elements in
    [0, len(int_mapped_docs)]
    :return:
        col_row: (batch_col_row) dict of the form:
            {'col': Torch Tensor; the padded and sorted-by-length sentence.
             'row': Torch Tensor; the padded and sorted-by-length entities.
             'col_lens': list(int); lengths of all sequences in 'col'.
             'row_lens': list(int); lengths of all sequences in 'row'.
             'sorted_colrefs': list(int); ints saying which seq in col came
                    from which document. ints in range [0, len(docs)]
             'sorted_rowrefs': list(int); ints saying which seq in row came
                    from which document. ints in range [0, len(docs)]}
        row_neg: (batch_row_neg) dict of the form:
            {'row': Torch Tensor; the padded and sorted-by-length entities.
             'row_lens': list(int); lengths of all sequences in 'col'.
             'sorted_rowrefs': list(int); ints saying which seq in row came
                    from which document. ints in range [0, len(docs)]}
    """
    assert (len(im_col_raw) == len(im_row_raw))
    col, col_lens, sorted_colrefs = pad_sort_seq(im_col_raw)
    row, row_lens, sorted_rowrefs = pad_sort_seq(im_row_raw)
    col_row = {'col': col,
               'row': row,
               'col_lens': col_lens,
               'row_lens': row_lens,
               'sorted_colrefs': sorted_colrefs,
               'sorted_rowrefs': sorted_rowrefs}
    if im_row_raw_neg:
        assert (len(im_col_raw) == len(im_row_raw) == len(im_row_raw_neg))
        row_neg, row_neg_lens, sorted_neg_rowrefs = pad_sort_seq(im_row_raw_neg)
        row_neg = {'row': row_neg,
                   'row_lens': row_neg_lens,
                   'sorted_rowrefs': sorted_neg_rowrefs}
        return col_row, row_neg

    return col_row


class Batcher():
    def __init__(self, full_col, full_row, batch_size=None, train_mode=True):
        """
        Maintain batcher variables and state and such.
        :param full_row: the full dataset. the int-mapped preprocessed sentences.
        :param full_col: the labels for the full dataset.
        :param batch_size: the number of documents to have in a batch; so sentence
            count varies.
        :param train_mode: boolean; Behaviour changes from train to test time. Like
            in shuffling data and generating negative examples.
        """
        self.full_len = len(full_col)
        self.batch_size = batch_size if batch_size != None else self.full_len
        assert(self.full_len == len(full_row))
        if self.full_len > self.batch_size:
            self.num_batches = int(np.ceil(float(self.full_len)/self.batch_size))
        else:
            self.num_batches = 1
        self.train_mode = train_mode
        if self.train_mode:
            # Get random permutation of the indices.
            # https://stackoverflow.com/a/19307027/3262406
            rand_indices = range(self.full_len)
            random.shuffle(rand_indices)
            # Shuffle once when the class initialized and then keep it that way.
            self.full_col = [full_col[i] for i in rand_indices]
            self.full_row = [full_row[i] for i in rand_indices]
            # Shuffle one more time and get negative examples.
            rand_indices = range(self.full_len)
            random.shuffle(rand_indices)
            self.full_row_neg = [full_row[i] for i in rand_indices]
        else:
            self.full_col = full_col
            self.full_row = full_row
        # Get batch indices.
        self.batch_start = 0
        self.batch_end = self.batch_size

    def next_batch(self):
        """
        Return the next batch.
        :return:
        """
        for nb in xrange(self.num_batches):
            if self.batch_end < self.full_len:
                batch_col_raw = self.full_col[self.batch_start:self.batch_end]
                batch_row_raw = self.full_row[self.batch_start:self.batch_end]
                if self.train_mode:
                    batch_row_raw_neg = self.full_row_neg[self.batch_start:self.batch_end]
            else:
                batch_col_raw = self.full_col[self.batch_start:]
                batch_row_raw = self.full_row[self.batch_start:]
                if self.train_mode:
                    batch_row_raw_neg = self.full_row_neg[self.batch_start:]
            self.batch_start = self.batch_end
            self.batch_end += self.batch_size
            if self.train_mode:
                batch_cr, batch_neg = pad_sort_data(
                    im_col_raw=batch_col_raw, im_row_raw=batch_row_raw,
                    im_row_raw_neg=batch_row_raw_neg)
                yield batch_cr, batch_neg
            else:
                batch_cr = pad_sort_data(im_col_raw=batch_col_raw,
                                         im_row_raw=batch_row_raw)
                yield batch_cr

    def full_batch(self):
        """
        Return the full batch.
        :return:
        """
        raise NotImplementedError


if __name__ == '__main__':
    if sys.argv[1] == 'test_pad_sort':
        int_mapped_path = sys.argv[2]
        dev_path = os.path.join(int_mapped_path, 'dev_ent_added-im-small.json')
        with open(dev_path) as fp:
            col_rel, row_ent, doc_ids = json.load(fp)
        testX, testy = col_rel[5:9], row_ent[5:9]
        print(testX)
        print(testy)
        col_row = pad_sort_data(testX, testy)
        print(col_row)
    elif sys.argv[1] == 'test_batcher':
        int_mapped_path = sys.argv[2]
        dev_path = os.path.join(int_mapped_path, 'dev_ent_added-im-small.json')
        with open(dev_path) as fp:
            col_rel, row_ent, doc_ids = json.load(fp)
        testc, testr = col_rel[:10], row_ent[:10]
        test_batcher = Batcher(testc, testr, 3)
        for cr, neg in test_batcher.next_batch():
            print(cr['row'].size())
            print(cr['col'].size())
            print(neg['row'].size())
    elif sys.argv[1] == 'test_embed_init':
        embed_path = sys.argv[2]
        int_mapped_path = sys.argv[3]
        map_path = os.path.join(int_mapped_path, 'word2idx-small.json')
        with open(map_path, 'r') as fp:
            word2idx = json.load(fp)
        embeds = init_pretrained_embed(embed_path=embed_path,
                                       word2idx=word2idx, embedding_dim=100)
        print(embeds.num_embeddings, embeds.embedding_dim, embeds.padding_idx)
    else:
        sys.argv.write('Unknown argument.\n')