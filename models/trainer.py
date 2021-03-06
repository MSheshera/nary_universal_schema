"""
Train the passed model given the data and the batcher and save the best to disk.
"""
from __future__ import print_function
import sys, os
import time, copy

import numpy as np
from sklearn import metrics
import torch
import torch.optim as optim

import model_utils as mu


class LSTMTrainer():
    def __init__(self, model, data, batcher, batch_size, update_rule, num_epochs,
                 learning_rate, check_every, print_every, model_path,
                 verbose=True):
        """
        Trainer class that I can hopefully reuse with little changes.
        :param model: pytorch model.
        :param data: dict{'X_train':, 'y_train':, 'X_dev':, 'y_dev':}
        :param batcher: a model_utils.Batcher class.
        :param batch_size: int; number of docs to consider in a batch.
        :param update_rule: string;
        :param num_epochs: int; number of passes through the training data.
        :param learning_rate: float;
        :param check_every: int; check model dev_f1 check_every iterations.
        :param print_every: int; print training numbers print_every iterations.
        :param model_path: string; directory to which model should get saved.
        :param verbose: boolean;
        """
        # Model, batcher and the data.
        self.model = model
        self.batcher = batcher
        self.col_train = data['col_train']
        self.row_train = data['row_train']
        self.col_dev = data['col_dev']
        self.row_dev = data['row_dev']

        # Book keeping
        self.verbose = verbose
        self.print_every = print_every
        self.check_every = check_every
        self.num_train = len(self.col_train)
        self.num_dev = len(self.col_dev)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if self.num_train > self.batch_size:
            self.num_batches = int(np.ceil(float(self.num_train)/self.batch_size))
        else:
            self.num_batches = 1
        self.model_path = model_path  # Save model and checkpoints.
        self.total_iters = self.num_epochs*self.num_batches

        # Optimizer args.
        self.update_rule = update_rule
        self.learning_rate = learning_rate

        # Initialize optimizer.
        if self.update_rule == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate)
        else:
            sys.stderr.write('Unknown update rule.\n')
            sys.exit(1)

        # Train statistics.
        self.loss_history = []
        self.dev_score_history = []
        self.checked_iters = []

    def train(self):
        """
        Make num_epoch passes throught the training set and train the model.
        :return:
        """
        # Pick the model with the least loss.
        best_params = self.model.state_dict()
        best_epoch, best_iter = 0, 0
        best_dev_loss = np.inf

        train_start = time.time()
        print('num_train: {:d}; num_dev: {:d}'.format(self.num_train,
                                                      self.num_dev))
        print('Training {:d} epochs'.format(self.num_epochs))
        iter = 0
        for epoch in xrange(self.num_epochs):
            # Initialize batcher. Shuffle one time before the start of every
            # epoch.
            epoch_batcher = self.batcher(full_col=self.col_train,
                                         full_row=self.row_train,
                                         batch_size=self.batch_size,
                                         train_mode=True)
            # Get the next padded and sorted training batch.
            iters_start = time.time()
            for batch_cr, batch_neg in epoch_batcher.next_batch():
                # Clear all gradient buffers.
                self.optimizer.zero_grad()
                # Compute objective.
                objective = self.model.forward(batch_cr=batch_cr, batch_neg=batch_neg)
                # Gradients wrt the parameters.
                objective.backward()
                # Step in the direction of the gradient.
                self.optimizer.step()
                loss = float(objective.data)
                if self.verbose and iter % self.print_every == 0:
                    print('Epoch: {:d}; Iteration: {:d}/{:d}; Loss: {:.4f}'.format(
                        epoch, iter, self.total_iters, loss))
                # Check every few iterations how you're doing on the dev set.
                if iter % self.check_every == 0:
                    dev_loss = mu.batched_loss(
                        model=self.model, batcher=self.batcher, batch_size=64,
                        full_col=self.col_dev, full_row=self.row_dev)
                    self.dev_score_history.append(dev_loss)
                    self.loss_history.append(loss)
                    self.checked_iters.append(iter)
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        # Deep copy so you're not just getting a reference.
                        best_params = copy.deepcopy(self.model.state_dict())
                        best_epoch = epoch
                        best_iter = iter
                        everything = (epoch, iter, self.total_iters, dev_loss)
                        if self.verbose:
                            print('Current best model; Epoch {:d}; Iteration '
                                  '{:d}/{:d}; Dev loss: {:.4f}'.
                                  format(*everything))
                iter += 1
            epoch_time = time.time()-iters_start
            print('Epoch {:d} time: {:.4f}s'.format(epoch, epoch_time))
            print()

        # Update model parameters to be best params.
        print('Best model; Epoch {:d}; Iteration {:d}; Dev F1: {:.4f}'.
              format(best_epoch, best_iter, best_dev_loss))
        self.model.load_state_dict(best_params)
        train_time = time.time()-train_start
        print('Training time: {:.4f}s'.format(train_time))

        # Save the learnt model.
        # https://stackoverflow.com/a/43819235/3262406
        model_file = os.path.join(self.model_path, 'ncompus_best.pt')
        torch.save(self.model.state_dict(), model_file)
        print('Wrote: {:s}'.format(model_file))
