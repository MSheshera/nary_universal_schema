"""
Miscellaneous utilities to read and work with the json files and such.
Stuff multiple functions use.
"""
from __future__ import unicode_literals
from __future__ import print_function
import codecs, json, random


def read_perline_json(json_file):
    """
    Read per line JSON and yield.
    :param json_file: Just a open file. file-like with a next method.
    :return: yield one json object.
    """
    for json_line in json_file:
        # Try to manually skip bad chars.
        # https://stackoverflow.com/a/9295597/3262406
        try:
            f_dict = json.loads(json_line.replace('\r\n', '\\r\\n'),
                                encoding='utf-8')
            yield f_dict
        # Skip case which crazy escape characters.
        except ValueError:
            yield {}


def read_rawms_json(paper_path):
    """
    Read data needed from JSON the raw material science file specfied and
    return.
    :param paper_path: string; path to the paper to read from.
    :return:
        paper_doi: string
        sents: list(string); whole sentences for readability.
        sents_toks: list(list(str)); Tokenized sentences.
        sents_labs: list(list(str)); Labeles of tokenized sentences.
    """
    with codecs.open(paper_path, u'r', u'utf-8') as fp:
        paper_dict = json.load(fp, encoding=u'utf-8')

    sents = []
    sents_toks = []
    sents_labs = []
    paper_doi = paper_dict['doi']
    for par_dict in paper_dict['paragraphs']:
        sents.extend(par_dict['sentences_proc'])
        sents_toks.extend(par_dict['proc_tokens'])
        sents_labs.extend(par_dict['proc_labels_cnnner'])

    return paper_doi, sents, sents_toks, sents_labs


def get_rand_indices(maxnum):
    """
    Return a permutation of the [0:maxnum-1] range.
    :return:
    """
    indices = range(maxnum)
    # Get random permutation of the indices but control for randomness.
    # https://stackoverflow.com/a/19307027/3262406
    random.shuffle(indices, lambda: 0.4186)
    return indices