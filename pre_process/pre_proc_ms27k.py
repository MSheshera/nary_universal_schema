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
        pass
    else:
        sys.stderr.write('Unknown action.\n')


if __name__ == '__main__':
    main()