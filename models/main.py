"""
Call code from everywhere, read data, initialize model, train model and make
sure training is doing something meaningful.
"""
from __future__ import unicode_literals
from __future__ import print_function
import argparse, os, sys
import math, time, codecs, pprint, json

import torch
import utils
import model_utils as mu
import narycompus as ncompus
import trainer
import evaluate

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def train_model(int_mapped_path, embedding_path, run_path, model_hparams,
                train_hparams, use_toy=True):
    """
    Read the int training and dev data, initialize and train the model.
    :return:
    """
    # Load training and dev data.
    train, dev, test, word2idx = utils.load_intmapped_data(int_mapped_path,
                                                           use_toy)
    data = {'col_train': train['col_rel'],
            'row_train': train['row_ent'],
            'col_dev': dev['col_rel'],
            'row_dev': dev['row_ent']}
    # Unpack args.
    hdim, dropp = model_hparams['hdim'], model_hparams['dropp']
    bsize, epochs, lr = train_hparams['bsize'], train_hparams['epochs'], train_hparams['lr']
    print('Model hyperparams:')
    pprint.pprint(model_hparams)
    print('Train hyperparams:')
    pprint.pprint(train_hparams)

    # Initialize model.
    model = ncompus.NaryCompUSchema(word2idx, embedding_path, num_layers=1,
                                    hidden_dim=hdim, dropout=dropp)
    print(model)

    # Initialize the trainer.
    lstmtrainer = trainer.LSTMTrainer(
        model=model, data=data, batcher=mu.Batcher, batch_size=bsize,
        update_rule='adam', num_epochs=epochs, learning_rate=lr, check_every=100,
        print_every=100, model_path=run_path)

    # Train and save the best model to model_path.
    lstmtrainer.train()
    # Plot training time stats.
    utils.plot_train_hist(lstmtrainer.loss_history, lstmtrainer.checked_iters,
                          fig_path=run_path, ylabel='Batch loss')
    utils.plot_train_hist(lstmtrainer.dev_score_history,
                          lstmtrainer.checked_iters,
                          fig_path=run_path, ylabel='Dev-set Loss')

    # Evaluate on all the data.
    data['col_test'] = test['col_rel']
    data['row_test'] = test['row_ent']
    # Add doc ids in.
    data['doc_ids_train'] = train['doc_ids']
    data['doc_ids_dev'] = dev['doc_ids']
    data['doc_ids_test'] = test['doc_ids']
    probs_train, probs_dev, probs_test = evaluate.make_predictions(
        data=data, batcher=mu.Batcher, model=model, result_path=run_path,
        batch_size=128, write_preds=True)

    print('train'); print(probs_train[:10]); print(train['doc_ids'][:10])
    print('dev'); print(probs_dev[:10]); print(dev['doc_ids'][:10])
    print('test'); print(probs_test[:10]); print(test['doc_ids'][:10])

    # Save embeddings to disk.
    if torch.cuda.is_available():
        learnt_embedding = model.embeddings.cpu()
        learnt_embedding = learnt_embedding.weight.data.numpy()
    else:
        learnt_embedding = model.embeddings.weight.data.numpy()
    print('learnt embeddings shape: {}'.format(learnt_embedding.shape))
    evaluate.write_embeddings(embeddings=learnt_embedding, run_path=run_path)

    # Save hyperparams to disk.
    run_info = {'model_hparams': model_hparams,
                'train_hparams': train_hparams}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)


def run_saved_model(int_mapped_path, embedding_path, run_path, use_toy=True):
    """
    Read the int training and dev data, initialize and run a saved model.
    :return:
    """
    # Load training and dev data.
    train, dev, test, word2idx = utils.load_intmapped_data(int_mapped_path,
                                                           use_toy)

    data = {'col_train': train['col_rel'],
            'row_train': train['row_ent'],
            'col_dev': dev['col_rel'],
            'row_dev': dev['row_ent']}
    # Load the hyperparams from disk.
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'r', 'utf-8') as fp:
        run_info = json.load(fp)
        model_hparams = run_info['model_hparams']
    # Unpack args.
    hdim, dropp = model_hparams['hdim'], model_hparams['dropp']
    print('Model hyperparams:')
    pprint.pprint(model_hparams)

    # Initialize model.
    model = ncompus.NaryCompUSchema(word2idx, embedding_path, num_layers=1,
                                    hidden_dim=hdim, dropout=dropp)
    print(model)
    model_file = os.path.join(run_path, 'ncompus_best.pt')
    model.load_state_dict(torch.load(model_file))
    # Evaluate on all the data; write predicted probs, write learnt embeddings.
    data['col_test'] = test['col_rel']
    data['row_test'] = test['row_ent']
    # Add doc ids in.
    data['doc_ids_train'] = train['doc_ids']
    data['doc_ids_dev'] = dev['doc_ids']
    data['doc_ids_test'] = test['doc_ids']
    probs_train, probs_dev, probs_test = evaluate.make_predictions(
        data=data, batcher=mu.Batcher, model=model, result_path=run_path,
        batch_size=128, write_preds=True)

    # Save embeddings to disk.
    if torch.cuda.is_available():
        learnt_embedding = model.embeddings.cpu()
        learnt_embedding = learnt_embedding.weight.data.numpy()
    else:
        learnt_embedding = model.embeddings.weight.data.numpy()
    print('learnt embeddings shape: {}'.format(learnt_embedding.shape))
    evaluate.write_embeddings(embeddings=learnt_embedding, run_path=run_path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')
    # Train the model.
    train_args = subparsers.add_parser(u'train_model')
    # Where to get what.
    train_args.add_argument(u'--int_mapped_path', required=True,
                            help=u'Path to the int mapped dataset.')
    train_args.add_argument(u'--embedding_path', required=True,
                            help=u'Path to the glove embeddings directory.')
    train_args.add_argument(u'--run_path', required=True,
                            help=u'Path to directory to save all run items to.')
    # Model hyper-parameters.
    train_args.add_argument(u'--edim', required=True, type=int,
                            choices=[50, 100, 200, 300],
                            help=u'Embedding dimension.')
    train_args.add_argument(u'--hdim', required=True, type=int,
                            choices=range(1, 201, 1),
                            help=u'LSTM hidden dimension.')
    train_args.add_argument(u'--dropp', required=True, type=float,
                            choices=[0., 0.1, 0.2, 0.3, 0.4,
                                     0.5, 0.6, 0.7, 0.8, 0.9],
                            help=u'Dropout probability.')
    # Training hyper-parameters.
    train_args.add_argument(u'--bsize', required=True, type=int,
                            choices=[32, 64, 128], help=u'Batch size.')
    train_args.add_argument(u'--epochs', required=True, type=int,
                            choices=[1, 2, 3, 4, 5],
                            help=u'Number of training epochs.')
    train_args.add_argument(u'--lr', required=True, type=float,
                            choices=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                            help=u'Learning rate.')

    # Run a saved model.
    savedrun_args = subparsers.add_parser(u'run_saved_model')
    # Where to get what.
    savedrun_args.add_argument(u'--int_mapped_path', required=True,
                               help=u'Path to the int mapped dataset.')
    savedrun_args.add_argument(u'--embedding_path', required=True,
                               help=u'Path to the glove embeddings directory.')
    savedrun_args.add_argument(u'--run_path', required=True,
                               help=u'Path to directory with all run items.')
    cl_args = parser.parse_args()

    if cl_args.subcommand == 'train_model':
        model_hparams, train_hparams = {}, {}
        model_hparams['edim'], model_hparams['hdim'], model_hparams['dropp'] = \
            cl_args.edim, cl_args.hdim, cl_args.dropp
        train_hparams['bsize'], train_hparams['epochs'], train_hparams['lr'] = \
            cl_args.bsize, cl_args.epochs, cl_args.lr
        train_model(int_mapped_path=cl_args.int_mapped_path,
                    embedding_path=cl_args.embedding_path,
                    run_path=cl_args.run_path,
                    model_hparams=model_hparams, train_hparams=train_hparams,
                    use_toy=False)
    elif cl_args.subcommand == 'run_saved_model':
        run_saved_model(int_mapped_path=cl_args.int_mapped_path,
                        embedding_path=cl_args.embedding_path,
                        run_path=cl_args.run_path,
                        use_toy=False)

if __name__ == '__main__':
    main()
