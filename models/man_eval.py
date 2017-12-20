"""
Helper functions to help manual evaluation.
"""
from __future__ import unicode_literals
from __future__ import print_function

import os, sys, argparse
import codecs, json, time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import spatial

import utils

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def make_docid2doc(int_mapped_path):
    docid2doc = {}
    splits = ['dev_ent_added.json', 'test_ent_added.json', 'train_ent_added.json']
    for split in splits:
        split_file = os.path.join(int_mapped_path, split)
        with codecs.open(split_file, 'r', 'utf-8') as fp:
            print('Processing: {:s}'.format(split_file))
            for data_json in utils.read_json(fp):
                doc_id = data_json['doc_id']
                docid2doc[doc_id] = {'ents': data_json['ents'],
                                     'text': data_json['text']}
    docid2doc_file = os.path.join(int_mapped_path, 'docid2doc.json')
    print('docids2doc: {:d}'.format(len(docid2doc)))
    with codecs.open(docid2doc_file, 'w', 'utf-8') as fp:
        json.dump(docid2doc, fp)
        print('Wrote: {:s}'.format(fp.name))


def nearest_entities(int_mapped_path, run_path):
    """
    - Read entity embeddings.
    - Find nearest neighbours the entities.
    - Print the entities.
    :param int_mapped_path:
    :param run_path:
    :return:
    """
    # Read idx2word map.
    idx2word_file = os.path.join(int_mapped_path, 'word2idx-full.json')
    with codecs.open(idx2word_file, 'r', 'utf-8') as fp:
        word2idx = json.load(fp)
        print('Read: {}'.format(fp.name))
    idx2word = dict([(v, k) for k, v in word2idx.items()])
    print('word2idx: {}'.format(len(word2idx)))

    # Read docids2doc
    docids2doc_file = os.path.join(int_mapped_path, 'docid2doc.json')
    with codecs.open(docids2doc_file, 'r', 'utf-8') as fp:
        docids2doc = json.load(fp)
        print('Read: {}'.format(fp.name))

    # Read embeddings.
    with open(os.path.join(run_path, 'learnt_embeddings.npy'), 'r') as fp:
        embeddings = np.load(fp)
        print('Read: {}'.format(fp.name))
    print('Embeddings: {}'.format(embeddings.shape))

    # Get the entity embeddings out.
    unique_ents = set()
    for id, itemsdict in docids2doc.items():
        ents = itemsdict['ents']
        unique_ents.update(ents)
    print('Unique entities: {}'.format(len(unique_ents)))

    # Build a nearest neighbour search data structure.
    start = time.time()
    nearest_ents = NearestNeighbors(n_neighbors=11, metric='cosine')
    nearest_ents.fit(embeddings)
    end = time.time()
    print('Neighbour data structure formed in: {:.4f}s'.format(end - start))

    # For the entities print out the nearest entities.
    start = time.time()
    with codecs.open(os.path.join(run_path, 'entity_neighbors.txt'),'w', 'utf-8') as resfile:
        count = 0
        for ent in unique_ents:
            try: intid = word2idx[ent]
            except KeyError: print('This shouldnt have happened at all.')
            ent_vec = embeddings[intid, :]
            ent_vec = ent_vec.reshape(1, ent_vec.shape[0])
            neigh_ids = nearest_ents.kneighbors(ent_vec, return_distance=False)
            neighbours = [idx2word[id] for id in list(neigh_ids[0])]
            resfile.write(ent + '\n')
            # Omit itself from the list.
            resfile.write('\t'.join(neighbours[1:]))
            resfile.write('\n\n')
            count += 1
            if count > 10000:
                break
        print('Wrote results in: {:s}'.format(resfile.name))
    end = time.time()
    print('Nearest neighbours found in: {:.4f}s'.format(end - start))


def split_nearest_hidden(doc_ids, docids2doc, run_path, split_str, row_col_str):
    """
    - Read entity embeddings.
    - Find nearest neighbours the entities.
    - Print the entities.
    :param int_mapped_path:
    :param run_path:
    :return:
    """
    # Read embeddings.
    hidden_vecs_file = os.path.join(run_path, '{:s}_{:s}_hidden.npy'.
                               format(split_str, row_col_str))
    with open(hidden_vecs_file) as fp:
        hidden_vecs = np.load(fp)
        print('Read: {}'.format(fp.name))
    print('{:s} {:s} hidden vectors: {}'.format(split_str, row_col_str,
                                                hidden_vecs.shape))

    # Build a nearest neighbour search data structure.
    start = time.time()
    nearest_rc = NearestNeighbors(n_neighbors=6, metric='cosine')
    nearest_rc.fit(hidden_vecs)
    end = time.time()
    print('Neighbour data structure formed in: {:.4f}s'.format(end - start))

    # For the entities print out the nearest entities.
    key1 = 'text' if row_col_str == 'col' else 'ents'
    key2 = 'text' if row_col_str == 'row' else 'ents'
    start = time.time()
    resfilepath=os.path.join(run_path, '{:s}_{:s}_neighbors.txt'
                             .format(split_str, row_col_str))
    with codecs.open(resfilepath,'w', 'utf-8') as resfile:
        count = 0
        for doc_id_index, doc_id in enumerate(doc_ids):
            query_vec = hidden_vecs[doc_id_index, :]
            query_vec = query_vec.reshape(1, query_vec.shape[0])
            neigh_ids = nearest_rc.kneighbors(query_vec, return_distance=False)
            val1 = docids2doc[doc_id][key1]
            val2 = docids2doc[doc_id][key2]
            resfile.write('Query {:s}: {:s}\n'.format(key1, val1))
            resfile.write('Corresponding {:s}: {:s}\n'.format(key2, val2))
            # Omit itself from the list.
            neighbourdocids = [doc_ids[id] for id in list(neigh_ids[0])][1:]
            for i, neighbourdocid in enumerate(neighbourdocids):
                val1 = docids2doc[neighbourdocid][key1]
                val2 = docids2doc[neighbourdocid][key2]
                resfile.write('Neigh{:d} {:s}: {:s}\n'.format(i+1, key1, val1))
                resfile.write('Neigh{:d} {:s}: {:s}\n'.format(i+1, key2, val2))
            resfile.write('\n\n')
            count += 1
            if count > 3000:
                break
        print('Wrote results in: {:s}'.format(resfile.name))
    end = time.time()
    print('Nearest neighbours found in: {:.4f}s'.format(end - start))
    print()


def nearest_hidden(int_mapped_path, run_path, row_col_str):
    # Read docids2doc
    docids2doc_file = os.path.join(int_mapped_path, 'docid2doc.json')
    with codecs.open(docids2doc_file, 'r', 'utf-8') as fp:
        docids2doc = json.load(fp)
        print('Read: {}'.format(fp.name))
    # Get the docids.
    train, dev, test, word2idx = utils.load_intmapped_data(int_mapped_path,
                                                           use_toy=False)
    for split_str, split in [('train', train), ('dev', dev), ('test', test)]:
        doc_ids = split['doc_ids']
        split_nearest_hidden(doc_ids, docids2doc, run_path, split_str,
                             row_col_str)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')
    # Nearest ents.
    near_ents = subparsers.add_parser(u'nearest_ents')
    # Where to get what.
    near_ents.add_argument(u'--int_mapped_path', required=True,
                            help=u'Path to the int mapped dataset.')
    near_ents.add_argument(u'--run_path', required=True,
                           help=u'Path to directory with all run items.')
    # Nearest entitiy tuples
    near_rows = subparsers.add_parser(u'nearest_row')
    # Where to get what.
    near_rows.add_argument(u'--int_mapped_path', required=True,
                           help=u'Path to the int mapped dataset.')
    near_rows.add_argument(u'--run_path', required=True,
                           help=u'Path to directory with all run items.')

    # Nearest surface text.
    near_cols = subparsers.add_parser(u'nearest_col')
    near_cols.add_argument(u'--int_mapped_path', required=True,
                           help=u'Path to the int mapped dataset.')
    near_cols.add_argument(u'--run_path', required=True,
                           help=u'Path to directory with all run items.')
    cl_args = parser.parse_args()

    if cl_args.subcommand == 'nearest_ents':
        nearest_entities(int_mapped_path=cl_args.int_mapped_path,
                         run_path=cl_args.run_path)
    elif cl_args.subcommand == 'nearest_row':
        nearest_hidden(int_mapped_path=cl_args.int_mapped_path,
                       run_path=cl_args.run_path, row_col_str='row')
    elif cl_args.subcommand == 'nearest_col':
        nearest_hidden(int_mapped_path=cl_args.int_mapped_path,
                       run_path=cl_args.run_path, row_col_str='col')

if __name__ == '__main__':
    if sys.argv[1] == 'make_docid2doc':
        int_mapped_path = sys.argv[2]
        make_docid2doc(int_mapped_path)
    else:
        main()