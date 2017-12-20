"""
Code to make predictions and evaluate.
"""
from __future__ import print_function
import argparse, os, sys
import math, time, json, codecs, pickle

from sklearn import metrics
import numpy as np

import utils
import model_utils as mu


def evaluate_preds(true_y, pred_y):
    """
    Helper to compute eval metrics.
    :param true_y: numpy array; (num_samples, )
    :param pred_y: numpy array; (num_samples, )
    :return: wf1, wp, wr, ac; floats
    """
    wf1 = metrics.f1_score(true_y, pred_y, average='weighted')
    wp = metrics.precision_score(true_y, pred_y, average='weighted')
    wr = metrics.recall_score(true_y, pred_y, average='weighted')
    ac = metrics.accuracy_score(true_y, pred_y)
    print('Weighted F1: {:.4f}'.format(wf1))
    print('Weighted precision: {:.4f}'.format(wp))
    print('Weighted recall: {:.4f}'.format(wr))
    print('Accuracy: {:.4f}'.format(ac))
    print(metrics.classification_report(y_true=true_y, y_pred=pred_y))
    print()
    return wf1, wp, wr, ac


def make_predictions(data, batcher, model, result_path, batch_size,
                     write_preds=False):
    """
    Make predictions on passed data in batches with the model and save if asked.
    :param model: pytorch model.
    :param data: dict{'X_train':, 'y_train':, 'X_dev':, 'y_dev':,
        'X_test':, 'y_test':}
    :param batcher: reference to model_utils.Batcher class.
    :param result_path: the path to which predictions should get written.
    :param batch_size: int; number of docs to consider in a batch.
    :param write: if predictions should be written.
    :return: Return in order truth and prediction.
        y_test_true, y_test_preds: numpy array; (num_samples, )
        y_dev_true, y_dev_preds: numpy array; (num_samples, )
        y_train_true, y_train_preds: numpy array; (num_samples, )
    """
    # Unpack data.
    col_train, row_train, col_dev, row_dev, col_test, row_test = \
        data['col_train'], data['row_train'], data['col_dev'], \
        data['row_dev'], data['col_test'], data['row_test']
    doc_ids_train, doc_ids_dev, doc_ids_test = \
        data['doc_ids_train'], data['doc_ids_dev'], data['doc_ids_test']
    # Make predictions on the test, dev and train sets.

    start = time.time()
    probs_test, col_hidden_test, row_hidden_test = mu.batched_predict(
        model, batcher, batch_size, full_col=col_test, full_row=row_test)
    print('Test prediction time: {:.4f}s'.format(time.time()-start))

    start = time.time()
    probs_dev, col_hidden_dev, row_hidden_dev = mu.batched_predict(
        model, batcher, batch_size, full_col=col_dev, full_row=row_dev)
    print('Dev prediction time: {:.4f}s'.format(time.time() - start))

    start = time.time()
    probs_train, col_hidden_train, row_hidden_train = mu.batched_predict(
        model, batcher, batch_size, full_col=col_train, full_row=row_train)
    print('Train prediction time: {:.4f}s'.format(time.time() - start))

    if write_preds:
        test_preds_file = os.path.join(result_path, 'test')
        write_predictions(test_preds_file, doc_ids_test, probs_test,
                          col_hidden_test, row_hidden_test)

        dev_preds_file = os.path.join(result_path, 'dev')
        write_predictions(dev_preds_file, doc_ids_dev, probs_dev,
                          col_hidden_dev, row_hidden_dev)

        train_preds_file = os.path.join(result_path, 'train')
        write_predictions(train_preds_file, doc_ids_train, probs_train,
                          col_hidden_train, row_hidden_train)

    return probs_train, probs_dev, probs_test


def write_predictions(pred_file, doc_ids, probs, col_hidden, row_hidden):
    pred_dict = {}
    with codecs.open(pred_file+'_probs.json', 'w', 'utf-8') as fp:
        for i, doc_id in enumerate(doc_ids):
            pred_dict[doc_id] = {'prob': float(probs[i])}
        json.dump(pred_dict, fp)
        print('Wrote: {}'.format(fp.name))
    with open(pred_file + '_row_hidden.npy', 'w') as fp:
        np.save(fp, row_hidden)
        print('Wrote: {}'.format(fp.name))
    with open(pred_file + '_col_hidden.npy', 'w') as fp:
        np.save(fp, col_hidden)
        print('Wrote: {}'.format(fp.name))


def write_embeddings(embeddings, run_path):
    with open(os.path.join(run_path, 'learnt_embeddings.npy'), 'w') as fp:
        np.save(fp, embeddings)
        print('Wrote: {}'.format(fp.name))

if __name__ == '__main__':
    sys.stderr.write('Nothing to run.\n')