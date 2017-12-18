"""
Code to pre-process text data and embeddings to be fed to the model.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys, argparse
import codecs, json
import time

import utils

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def make_word2mse_dict(msew2v_dir):
    """
    Make a dict going from word to embeddings so its easy to read in later.
    :param msew2v_dir: path to the material science embeddings.
    :return: None.
    """
    word2mse = {}
    mse_file = os.path.join(msew2v_dir, 'msew2v.100d.txt')
    out_mse_file = os.path.join(msew2v_dir, '100d.embed.json')
    if os.path.isfile(out_mse_file):
        print('File exists; Look into it manually.\n')
        sys.exit(1)
    print('Processing: {:s}'.format(mse_file))
    skipped_lines = 0
    with codecs.open(mse_file, 'r', 'utf-8') as fp:
        for i, line in enumerate(fp):
            line = line.strip()
            if line:
                ss = line.split()
                word = ss[0]
                embeds = [float(x) for x in ss[1:]]
                emb_size = len(embeds)
                # Dont even know why this is happening; why are there lines with
                # no words. Screw it for now.
                if len(embeds) != 100:
                    skipped_lines += 1
                    continue
                if i % 1000 == 0:
                    try:
                        print('Processed: {:d} lines. Word: {:s}'.format(i, word))
                    # This shouldn't happen dk why it does.
                    except UnicodeDecodeError:
                        pass
                word2mse[word] = embeds
            else:
                skipped_lines += 1
    print('msew2v map length: {:d}'.format(len(word2mse)))
    print('Embedding dimension: {:d}'.format(emb_size))
    print('Empty lines: {:d}'.format(skipped_lines))
    # Save giant map to a json file.
    with codecs.open(out_mse_file, 'w', 'utf-8') as fp:
        json.dump(word2mse, fp)
    print('Wrote: {:s}'.format(out_mse_file))


def make_word2glove_dict(glove_dir):
    """
    Make a dict going from word to embeddings so its easy to read in later.
    :param glove_dir:
    :return:
    """
    word2glove = {}
    glove_file = os.path.join(glove_dir, 'glove.6B.100d.txt')
    out_glove_file = os.path.join(glove_dir, '100d.embed.json')
    if os.path.isfile(out_glove_file):
        print('File exists; Look into it manually.\n')
        sys.exit(1)
    print('Processing: {:s}'.format(glove_file))
    with codecs.open(glove_file, 'r', 'utf-8') as fp:
        for i, line in enumerate(fp):
            ss = line.split()
            word = ss[0]
            if i % 10000 == 0:
                print('Processed: {:d} lines. Word: {:s}'.format(i, word))
            embeds = [float(x) for x in ss[1:]]
            word2glove[word] = embeds
    print('w2g map length: {:d}'.format(len(word2glove)))
    # Save giant map to a json file.
    with codecs.open(out_glove_file, 'w', 'utf-8') as fp:
        json.dump(word2glove, fp)
    print('Wrote: {:s}'.format(out_glove_file))


def map_split_to_int(split_file, size_str, word2idx={}, update_map=True):
    """
    Convert text to set of int mapped tokens. Mapping words to integers at
    all times, train/dev/test. This should work transparently with
    grec and ms27k. grec is lowercased in toks and ents but not ms27k.
    :param split_file:
    :param word2idx:
    :param update_map:
    :return:
    """
    col_rel, row_ent, doc_ids = [], [], []
    num_tok_oovs, num_ent_oovs, num_docs = 0, 0, 0
    tok_vocab, ent_vocab = 0, 0
    # reserve index 0 for 'padding' on the ends and index 1 for 'oov'
    if '<pad>' not in word2idx:
        word2idx['<pad>'] = 0
    if '<oov>' not in word2idx:
        # reserve
        word2idx['<oov>'] = 1

    start_time = time.time()
    with codecs.open(split_file, 'r', 'utf-8') as fp:
        print('Processing: {:s}'.format(split_file))
        for data_json in utils.read_json(fp):
            # Read the list of (possibly multiword) entities.
            sent_ents = data_json['ents']
            # Read the sentence which you can just tokenize on white space.
            sent = data_json['text']
            # Keep track of the doc id because I think at some point I need to
            # look at the original text.
            doc_id = data_json['doc_id']
            num_docs += 1
            if num_docs % 10000 == 0:
                print('Processing {:d}th document (sentence)'.format(num_docs))
            # Make a small dataset if asked for.
            if size_str == 'small' and num_docs % 1000 == 0:
                break
            toks = sent.split()
            # Add start and stop states.
            toks = ['<start>'] + toks + ['<stop>']
            sent_ents = ['<start>'] + sent_ents + ['<stop>']
            if update_map:
                for tok in toks:
                    if tok not in word2idx:
                        tok_vocab += 1
                        word2idx[tok] = len(word2idx)
                # This is the point where if there were entity linking then
                # the different entity mentions would share one common integer
                # Here the linking is just perfect string matches.
                for sent_ent in sent_ents:
                    if sent_ent not in word2idx:
                        ent_vocab += 1
                        word2idx[sent_ent] = len(word2idx)
            # Map sentence tokens to integers. (these things also have the
            # representations for entity types)
            intmapped_sent = []
            for tok in toks:
                # This case cant happen for me because im updating the map
                # for every split. But in case I set update_map to false
                # this handles it. (Hopefully.)
                intmapped_tok = word2idx.get(tok, word2idx['<oov>'])
                intmapped_sent.append(intmapped_tok)
                if intmapped_tok == 1:
                    num_tok_oovs += 1
            # Map entities to integers.
            intmapped_sent_ents = []
            for sent_ent in sent_ents:
                intmapped_sent_ent = word2idx.get(sent_ent, word2idx['<oov>'])
                intmapped_sent_ents.append(intmapped_sent_ent)
                if intmapped_sent_ent == 1:
                    num_ent_oovs += 1
            col_rel.append(intmapped_sent)
            row_ent.append(intmapped_sent_ents)
            doc_ids.append(doc_id)
    assert(len(col_rel) == len(row_ent) == len(doc_ids))
    print('Processed: num_documents: {:d}; total_vocab_size: {:d}; '
          'tok_vocab_size: {:d}; ent_vocab_size: {:d} num_tok_oovs: {:d};'
          ' num_ent_oovs: {:d}'.format(num_docs, len(word2idx), tok_vocab,
                                       ent_vocab, num_tok_oovs, num_ent_oovs))
    print('Took: {:4.4f}s'.format(time.time()-start_time))
    return col_rel, row_ent, doc_ids, word2idx


def make_int_maps(in_path, out_path, size_str):
    """
    For each split map all the tokens + entities to integers and create
    int maps.
    :param in_path: dir with files: 'train_ent_added', 'dev_ent_added',
        'test_ent_added'
    :param out_path: path to which int mapped files should get written.
    :param size_str: says if you want to make a small 1000 example set or full.
    :return: None.
    """
    splits = ['train_ent_added', 'dev_ent_added', 'test_ent_added']
    word2idx = {}
    for split in splits:
        split_file = os.path.join(in_path, split) + '.json'
        col_rel, row_ent, doc_ids, word2idx = map_split_to_int(
            split_file, size_str=size_str, word2idx=word2idx, update_map=True)
        intmapped_out_file = os.path.join(out_path, split) + '-im-{:s}.json'.format(size_str)
        with codecs.open(intmapped_out_file, 'w', 'utf-8') as fp:
            json.dump((col_rel, row_ent, doc_ids), fp)
        print('Wrote: {:s}'.format(intmapped_out_file))
    # Write the map.
    intmap_out_path = os.path.join(out_path, 'word2idx-{:s}.json'.format(size_str))
    with codecs.open(intmap_out_path, 'w', 'utf-8') as fp:
        json.dump(word2idx, fp)
    print('Wrote: {:s}'.format(intmap_out_path))


def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')

    # Make the giant glove map.
    make_w2e = subparsers.add_parser(u'w2e_map')
    make_w2e.add_argument(u'-i', u'--embeddings_path',
                          required=True,
                          help=u'Path to the embeddings directory.')
    make_w2e.add_argument(u'-d', u'--dataset',
                          required=True, choices=['grec', 'ms27k'],
                          help=u'The dataset to process.')
    # Map sentences to list of int mapped tokens.
    make_int_map = subparsers.add_parser(u'int_map')
    make_int_map.add_argument(u'-i', u'--in_path', required=True,
                              help=u'Path to the processed train/dev/test '
                                   u'splits.')
    make_int_map.add_argument(u'-s', u'--size', required=True,
                              choices=['small', 'full'],
                              help=u'Make a small version with 1000 examples or'
                                   u'a large version with the whole dataset.')
    cl_args = parser.parse_args()

    if cl_args.subcommand == 'w2e_map':
        if cl_args.dataset == 'grec':
            make_word2glove_dict(glove_dir=cl_args.embeddings_path)
        elif cl_args.dataset == 'ms27k':
            make_word2mse_dict(msew2v_dir=cl_args.embeddings_path)
        else:
            pass
    if cl_args.subcommand == 'int_map':
        make_int_maps(in_path=cl_args.in_path, out_path=cl_args.in_path,
                      size_str=cl_args.size)


if __name__ == '__main__':
    main()