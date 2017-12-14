from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import codecs, json, gzip
import collections
import pprint
import re

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

wikibio_base_path = '/iesl/canvas/smysore/nary_rel_extract/datasets/WikiBio'
proc_wikibio_path = '/iesl/canvas/smysore/nary_rel_extract/processed/WikiBio'
parsedwiki_path = '/iesl/data/parsed_wikipedia/case_sensitive'
parsedwiki_subset_path = '/iesl/canvas/smysore/nary_rel_extract/processed/parsed_wikipedia_subset'


def make_wikibio_title_map():
    """
    Make a map of the titles in WikiBio; also store number of sentences in
    article. So I get only that many sentences from parsed_wiki.
    """
    splits = ['test', 'dev', 'train']
    for split in splits:
        title_to_id = {}
        title_file = codecs.open(os.path.join(wikibio_base_path, split, '{:s}.title'.format(split)),'r', 'utf-8')
        id_file = codecs.open(os.path.join(wikibio_base_path, split, '{:s}.id'.format(split)), 'r', 'utf-8')
        nb_file = codecs.open(os.path.join(wikibio_base_path, split, '{:s}.nb'.format(split)),'r', 'utf-8')
        split_map_file = codecs.open(os.path.join(proc_wikibio_path, '{:s}_title2id.json'.format(split)), 'w', 'utf-8')
        count = 0
        for title, article_id, sentences_nb in zip(title_file, id_file, nb_file):
            # Remove ends whitespace.
            title = title.strip()
            article_id = article_id.strip()
            sentences_nb = sentences_nb.strip()
            # Replace commas with a space before to no space before.
            cleaned_title = re.sub(r'\s,', ',', title)
            # Replace -lrb-, -rrb- with ( and ).
            cleaned_title = re.sub(r'-lrb-\s', '(', cleaned_title)
            cleaned_title = re.sub(r'\s-rrb-', ')', cleaned_title)
            # Replace all spaces with underscores.
            cleaned_title = re.sub(r'\s', '_', cleaned_title)
            # Store article and number of sentences so I can get it from that
            # many lines from the pased wiki file.
            title_to_id[cleaned_title] = [article_id, sentences_nb]
            count += 1
            if count % 1000 == 0:
                print('count: {:d}; cleaned_title: {:s}; title: {:s}'.format(count, cleaned_title, title))
        # Write out the map file.
        json.dump(title_to_id, split_map_file)
        title_file.close(); id_file.close(); split_map_file.close()
        print('Wrote: {:s}'.format(split_map_file.name))


def make_parsedwiki_subset():
    """
    Get the subset of the parsed wiki corpus which has the articles in Wikibio.
    :return:
    """
    with codecs.open(os.path.join(proc_wikibio_path, 'train_title2id.json'), 'r', 'utf-8') as fp:
        train_title2id = json.load(fp)
    print('Read train {:d} title2id'.format(len(train_title2id)))

    with codecs.open(os.path.join(proc_wikibio_path, 'dev_title2id.json'), 'r', 'utf-8') as fp:
        dev_title2id = json.load(fp)
    print('Read dev {:d} title2id'.format(len(dev_title2id)))

    with codecs.open(os.path.join(proc_wikibio_path, 'test_title2id.json'), 'r', 'utf-8') as fp:
        test_title2id = json.load(fp)
    print('Read test {:d} title2id'.format(len(test_title2id)))

    # Read lines as unicode so other stuff here works:
    # https://stackoverflow.com/a/1883734/3262406
    ureader = codecs.getreader('utf-8')
    fnames = os.listdir(parsedwiki_path)
    for fname in fnames:
        parsed_file = os.path.join(parsedwiki_path, fname)
        subset_file = os.path.join(parsedwiki_subset_path, fname.split('.')[0]) + '.tsv'
        subset_fp = codecs.open(subset_file, 'w', 'utf-8')
        with gzip.open(parsed_file, 'r') as zf:
            print(fname)
            zfcontents = ureader(zf)
            count = 0
            for line in zfcontents:
                line = line.strip()
                line_contents = line.split('\t')
                # First column is title and always says "Title:blah".
                title = (line_contents[0][6:]).strip()
                # Lowercase it because the WikiBio guys lowercased everything >_<
                norm_title = re.sub(r'\s', '_', title).lower()
                if (norm_title in train_title2id) or (norm_title in dev_title2id) or (norm_title in test_title2id):
                    subset_fp.write(line + '\n')
                    count += 1
                if count > 10:
                    break
        subset_fp.close()
        print('Wrote: {:s}'.format(subset_fp.name))


if __name__ == '__main__':
    # make_wikibio_title_map()
    # python2 -u explore_parsedwiki.py | tee ../../logs/title2id_logs.txt
    make_parsedwiki_subset()
    # python2 -u explore_parsedwiki.py | tee ../../logs/parsed_subset_logs.txt