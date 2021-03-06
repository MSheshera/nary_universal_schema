"""
General utilities; reading files and such.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys, glob
import itertools
import codecs, json, pprint

# Use mpl on remote.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def read_json(json_file):
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


def load_intmapped_data(int_mapped_path, use_toy):
    """
    Load the int mapped train, dev and test data from disk. Load a smaller
    dataset if use_toy is set.
    :param int_mapped_path:
    :param use_toy:
    :return: train, dev, test, word2idx:
        train/dev/test: dict('col_rel': list(list(str)),
            'row_ent': list(list(str)), 'doc_ids':list(str))
        word2ids: dict(str:int)
    """
    if use_toy:
        set_str = 'small'
    else:
        set_str = 'full'
    train_path = os.path.join(int_mapped_path,
                              'train_ent_added-im-{:s}.json'.format(set_str))
    with open(train_path) as fp:
        col_rel_tr, row_ent_tr, doc_ids_tr = json.load(fp)  # l of l.

    dev_path = os.path.join(int_mapped_path,
                            'dev_ent_added-im-{:s}.json'.format(set_str))
    with open(dev_path) as fp:
        col_rel_dev, row_ent_dev, doc_ids_dev = json.load(fp)  # l of l.

    test_path = os.path.join(int_mapped_path,
                             'test_ent_added-im-{:s}.json'.format(set_str))
    with open(test_path) as fp:
        col_rel_te, row_ent_te, doc_ids_te = json.load(fp)  # l of l.

    map_path = os.path.join(int_mapped_path, 'word2idx-{:s}.json'.format(set_str))
    with open(map_path) as fp:
        word2idx = json.load(fp)

    train = {'col_rel': col_rel_tr, 'row_ent': row_ent_tr, 'doc_ids':doc_ids_tr}
    dev = {'col_rel': col_rel_dev, 'row_ent': row_ent_dev, 'doc_ids': doc_ids_dev}
    test = {'col_rel': col_rel_te, 'row_ent': row_ent_te, 'doc_ids': doc_ids_te}
    return train, dev, test, word2idx


def plot_train_hist(y_vals, checked_iters, fig_path, ylabel):
    """
    Plot y_vals against the number of iterations.
    :param score: list
    :param loss: list; len(score)==len(loss)
    :param check_every: int
    :param fig_path: string;
    :return: None.
    """
    x_vals = np.array(checked_iters)
    y_vals = np.vstack(y_vals)
    plt.plot(x_vals, y_vals, '-', linewidth=2)
    plt.xlabel('Training iteration')
    plt.ylabel(ylabel)
    plt.title('Evaluated every: {:d} iterations'.format(
        checked_iters[1]-checked_iters[0]))
    plt.tight_layout()
    ylabel='_'.join(ylabel.lower().split())
    fig_file = os.path.join(fig_path, '{:s}_history.eps'.format(ylabel))
    plt.savefig(fig_file)
    plt.savefig(os.path.join(fig_path, '{:s}_history.png'.format(ylabel)))
    plt.clf()
    print('Wrote: {:s}'.format(fig_file))


if __name__ == '__main__':
    if sys.argv[1] == 'test_plot_hist':
        plot_train_hist([1,2,3,4], checked_iters=[100,200,300,400],
                        fig_path=sys.argv[2], ylabel='test')
    else:
        sys.argv.write('Unknown argument.\n')