"""
Pre-process the material science data for the nary-uschema task.
- Make splits of the data at the document level. 80-10-10.
- Process the splits to be of the form needed for naryus.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import codecs, json
import argparse
import re, copy, pprint
import time
from collections import defaultdict

import data_utils as du

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


###################################################################
#            Create splits at the document level.                 #
###################################################################
def make_mstdt_split(raw_ms27k_path):
    """
    Given the path to the raw data return file names to use for train,
    dev and test.
    :param raw_ms27k_path:
    :return:
        train_fnames: list(string)
        dev_fnames: list(string)
        test_fnames: list(string)
    """
    doc_fnames = os.listdir(raw_ms27k_path)
    num_examples = len(doc_fnames)
    # Get a permutation of the [0:num_examples-1] range.
    indices = du.get_rand_indices(num_examples)
    train_i = indices[:int(0.8 * num_examples)]
    dev_i = indices[int(0.8 * num_examples): int(0.9 * num_examples)]
    test_i = indices[int(0.9 * num_examples):]
    train_fnames = [doc_fnames[i] for i in train_i]
    dev_fnames = [doc_fnames[i] for i in dev_i]
    test_fnames = [doc_fnames[i] for i in test_i]
    return train_fnames, dev_fnames, test_fnames


def control_mstdt_split(raw_ms27k_path, split_path):
    """
    Get split filenames, read relevant info from the files and write to a single
    train, test and dev file.
    :param raw_ms27k_path:
    :return:
    """
    train_fnames, dev_fnames, test_fnames = make_mstdt_split(raw_ms27k_path)
    print('Train: {}; Dev: {}; Test: {}'.format(len(train_fnames),
                                                len(dev_fnames),
                                                len(test_fnames)))
    for split_str, split_fnames in [('train', train_fnames),
                                    ('dev', dev_fnames), ('test', test_fnames)]:
        start = time.time()
        print('Processing: {:s}'.format(split_str))
        split_fname = os.path.join(split_path, split_str) + '.json'
        split_file = codecs.open(split_fname, u'w', u'utf-8')
        split_sents_count = 0
        processed_fcount = 0
        for fname in split_fnames:
            paper_json_fname = os.path.join(raw_ms27k_path, fname)
            paper_doi, sents, sents_toks, sents_labs = du.read_rawms_json(
                paper_path=paper_json_fname)
            split_sents_count += len(sents)
            # TODO: Keep track of the keyword queries that returned this paper.
            paper_dict = {
                'paper_doi': paper_doi,
                'sents': sents,
                'sents_toks': sents_toks,
                'sents_labs': sents_labs
            }
            proc_jsons = json.dumps(paper_dict, ensure_ascii=False)
            split_file.write(proc_jsons + '\n')
            processed_fcount += 1
            if processed_fcount % 1000 == 0:
                print('Processed files: {:d}; Processing: {:s}'.format(processed_fcount, paper_doi))
        split_file.close()
        print('Split: {:s}; Sentences: {:d}; Sentences per doc: {:.4f}'.format(
            split_str, split_sents_count,
            float(split_sents_count)/processed_fcount))
        print('Wrote: {:s}'.format(split_fname))
        print('Took: {:.4f}s'.format(time.time()-start))
        print()


###################################################################
#     Swap entity tags to the sentences and get sent entities     #
###################################################################
def swap_labs_ents(sent_toks, sent_labs):
    """
    Swap the entity labels into the sentence and get the entities out of the
    sentence.
    :param sent_toks: list(str)
    :param sent_labs: list(str)
    :return:
    """
    assert(len(sent_toks) == len(sent_labs))
    ent_swapped_sent, sent_ents = [], []
    cur_lab = ''
    # Fix the predicted labels for obvious mistakes instead of trying to crazily
    # handle for it below. Fix stray I-tags without B-tags.
    inlab = False
    for i, lab in enumerate(sent_labs):
        if lab[0] == 'O':
            inlab = False
        if lab[0] == 'B':
            inlab = True
        if lab[0] == 'I' and inlab == True:
            continue
        elif lab[0] == 'I' and inlab == False:
            sent_labs[i] = 'B' + lab[1:]
            inlab = True

    for tok, lab in zip(sent_toks, sent_labs):
        if lab == 'O':
            ent_swapped_sent.append(tok)
        # You saw a beginning, so add ent to the ents and place label into
        # sentence.
        elif lab[0] == 'B':
            cur_lab = lab[2:]
            ent_swapped_sent.append(cur_lab + '_netag')
            sent_ents.append(tok)
        # You see a "in", if its the same current label concat it with the B ent
        # token and skip any addition to the sentence.
        elif lab[0] == 'I' and lab[2:] == cur_lab:
            sent_ents[-1] = (sent_ents[-1] + ' ' + tok).strip()
    return ent_swapped_sent, sent_ents


def process_ms27k_doc(data_dict):
    """
    For a given document, get the sentences which have an operation, replace the
    entities in the sentence with entitiy tags and get the entities.
    :param data_dict: dict('paper_doi': str, 'sents': list(str),
        'sents_toks': list(list(str)), 'sents_labs': list(list(str)))
    :return:
    """
    ent_swapped_sents = []
    sents_ents = []
    sents_toks = data_dict['sents_toks']
    sents_labs = data_dict['sents_labs']
    assert (len(sents_toks) == len(sents_labs))

    doi_str = data_dict['paper_doi']
    for sent_toks, sent_labs in zip(sents_toks, sents_labs):
        # Look at only operation sentences. This might be a bad judgement call.
        if ('B-operation' in sent_labs) or ('I-operation' in sent_labs):
            ent_swapped_sent, sent_ents = \
                swap_labs_ents(sent_toks=sent_toks, sent_labs=sent_labs)
            ent_swapped_sents.append(ent_swapped_sent)
            sents_ents.append(sent_ents)
    return doi_str, ent_swapped_sents, sents_ents


def process_ms27k_splits(split_path, out_path):
    """
    Make the splits into sentence level examples suited to the naryus task.
    :param split_path:
    :param out_path:
    :return:
    """
    for split_str in ['dev', 'test', 'train']:
        in_split_fname = os.path.join(split_path, split_str) + '.json'
        out_split_fname = os.path.join(out_path, split_str) + '_ent_added.json'
        out_split_file = codecs.open(out_split_fname, u'w', u'utf-8')
        print('Processing: {:s}'.format(in_split_fname))
        start = time.time()
        sents_count = 0
        ents_count = 0
        with codecs.open(in_split_fname, 'r', 'utf-8') as fp:
            for data_dict in du.read_perline_json(fp):
                doi_str, ent_swapped_sents, sents_ents = process_ms27k_doc(data_dict)
                for sent_id, (ent_swapped_sent, sent_ents) in enumerate(zip(ent_swapped_sents, sents_ents)):
                    out_dict = {
                        'text': ' '.join(ent_swapped_sent),
                        'ents': sent_ents,
                        'doc_id': '{:s}_{:d}'.format(doi_str, sent_id)
                    }
                    sents_count += 1
                    ents_count += len(sent_ents)
                    proc_jsons = json.dumps(out_dict, ensure_ascii=False)
                    out_split_file.write(proc_jsons + '\n')
                if sents_count % 1000 == 0:
                    print('Cur sents count: {:d}; Average ents per sent: {:f}'.
                          format(sents_count,float(ents_count) / sents_count))
            print('Total sents count: {:d}; Average ents per sent: {:f}'.
                  format(sents_count, float(ents_count) / sents_count))
            out_split_file.close()
            end = time.time()
            print('Wrote: {:s}'.format(out_split_fname))
            print('Took: {:.4f}s; Per sentence: {:.4f}s'.
                  format(end - start, float(end - start) / sents_count))
            print()


def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')

    # Action to make the general train/dev/test split.
    make_tdt_split = subparsers.add_parser(u'split')
    make_tdt_split.add_argument(u'-i', u'--in_path',
                                required=True,
                                help=u'The MS27 raw dataset path.')
    make_tdt_split.add_argument(u'-o', u'--out_path',
                                required=True,
                                help=u'Path to which splits should get written.')

    # Action to make the sentence level dataset for nary us.
    make_naryus_dataset = subparsers.add_parser(u'naryus')
    make_naryus_dataset.add_argument(u'-i', u'--in_path', required=True,
                                     help=u'Directory with the train/dev/test '
                                          u'split jsons.')
    make_naryus_dataset.add_argument(u'-o', u'--out_path', required=True,
                                     help=u'Directory to which the processed '
                                          u'splits, should be written.')
    cl_args = parser.parse_args()

    if cl_args.subcommand == 'split':
        # Dont want to overwrite existing files by same name. :/
        assert(cl_args.in_path != cl_args.out_path)
        control_mstdt_split(raw_ms27k_path=cl_args.in_path,
                            split_path=cl_args.out_path)
    elif cl_args.subcommand == 'naryus':
        # Dont want to overwrite existing files by same name. :/
        assert (cl_args.in_path != cl_args.out_path)
        process_ms27k_splits(split_path=cl_args.in_path,
                             out_path=cl_args.out_path)
    else:
        sys.stderr.write('Unknown action.\n')


if __name__ == '__main__':
    if sys.argv[1] != 'test':
        main()
    else:
        sent_toks = ["Lead", "titanate", "precursor", "gels", "were",
                     "prepared", "using", "a", "diol", "sol", "-", "gel",
                     "system", "."]
        sent_labs = ["I-material", "I-material", "B-descriptor", "O", "O",
                     "O", "O", "O", "O", "I-meta", "I-meta", "I-meta",
                     "I-meta", "O"]
        a, b = swap_labs_ents(sent_toks, sent_labs)
        print(a)
        print(b)
