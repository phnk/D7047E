"""
This file is the template for the boilerplate of train/test of a DNN for image classification

There are a lot of parameter which can be specified to modify the behaviour and they should be used 
instead of hard-coding stuff.
"""

import logging
import sys
import os

# Utils
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

# DeepDIVA
import models
# Delegated
from template.runner.process_activation import evaluate, train
from template.runner.process_activation.activation import Activation
from template.setup import set_up_model, set_up_dataloaders
from util.misc import checkpoint, adjust_learning_rate


class ProcessActivation:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr,
                   validation_interval, checkpoint_all_epochs, **kwargs):
        """
        DESC

        Parameters
        ----------
        Param
            Desc

        Returns
        -------
            None
        """

        if not kwargs['train'] and kwargs['load_model'] == None:
            logging.error('You have to provide load_model argument if model is not trained.')
            sys.exit(-1)

        # Get the selected model input size
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        ProcessActivation._validate_model_input_size(model_expected_input_size, model_name)
        logging.info('Model {} expects input size of {}'.format(model_name,
                                                                model_expected_input_size))

        # Setting up the dataloaders
        train_loader, val_loader, test_loader, num_classes = set_up_dataloaders(
            model_expected_input_size,
            **kwargs)

        # Freezing the dataset used for processing activation
        activation_dataset = []
        for i, data in enumerate(train_loader):
            activation_dataset.append(data)
            if i >= kwargs['process_size']:
                break

        # Setting up model, optimizer, criterion
        model, criterion, optimizer, best_value, start_epoch = set_up_model(num_classes=num_classes,
                                                                            model_name=model_name,
                                                                            lr=lr,
                                                                            train_loader=train_loader,
                                                                            **kwargs)

        # Setting up activation_worker
        activation_worker = Activation(current_log_folder,
                                       model_name,
                                       activation_dataset,
                                       kwargs['process_size'],
                                       kwargs['save_images'],
                                       kwargs['no_cuda'])
        activation_worker.init(model)
        activation_worker.resolve_items()

        # With training part
        if kwargs['train']:
            logging.info('Begin training')
            val_value = np.zeros((epochs + 1 - start_epoch))
            train_value = np.zeros((epochs - start_epoch))

            # Pretraining validation step
            val_value[-1] = ProcessActivation._validate(val_loader, model, criterion, writer, -1, **kwargs)

            # Training
            for epoch in range(start_epoch, epochs):
                train_value[epoch] = ProcessActivation._train(train_loader, model, criterion, optimizer, writer, epoch,
                                                              **kwargs)

                # Validate
                if epoch % validation_interval == 0:
                    val_value[epoch] = ProcessActivation._validate(val_loader, model, criterion, writer, epoch,
                                                                   **kwargs)

                # Activation
                if (epoch == start_epoch) or (epoch % kwargs['process_every'] == 0) or epoch == (epochs - 1):
                    activation_worker.add_epoch(epoch, val_value[epoch], model)

                if decay_lr is not None:
                    adjust_learning_rate(lr=lr, optimizer=optimizer, epoch=epoch, decay_lr_epochs=decay_lr)
                best_value = checkpoint(epoch=epoch, new_value=val_value[epoch],
                                        best_value=best_value, model=model,
                                        optimizer=optimizer,
                                        log_dir=current_log_folder,
                                        checkpoint_all_epochs=checkpoint_all_epochs)

            # Load the best model before evaluating on the test set.
            logging.info('Loading the best model before evaluating on the test set.')
            kwargs["load_model"] = os.path.join(current_log_folder, 'model_best.pth.tar')
            model, _, _, _, _ = set_up_model(num_classes=num_classes,
                                             model_name=model_name,
                                             lr=lr,
                                             train_loader=train_loader,
                                             **kwargs)

            # Test
            test_value = ProcessActivation._test(test_loader, model, criterion, writer, epochs - 1, **kwargs)
            logging.info('Training completed')

        # Without training part
        else:
            activation_worker.add_epoch(0, 0, model)

        sys.exit(-1)

    ####################################################################################################################
    @staticmethod
    def _validate_model_input_size(model_expected_input_size, model_name):
        """
        This method verifies that the model expected input size is a tuple of 2 elements.
        This is necessary to avoid confusion with models which run on other types of data.

        Parameters
        ----------
        model_expected_input_size
            The item retrieved from the model which corresponds to the expected input size
        model_name : String
            Name of the model (logging purpose only)

        Returns
        -------
            None
        """
        if type(model_expected_input_size) is not tuple or len(model_expected_input_size) != 2:
            logging.error('Model {model_name} expected input size is not a tuple. '
                          'Received: {model_expected_input_size}'
                          .format(model_name=model_name,
                                  model_expected_input_size=model_expected_input_size))
            sys.exit(-1)

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @classmethod
    def _train(cls, train_loader, model, criterion, optimizer, writer, epoch, **kwargs):
        return train.train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)

    @classmethod
    def _validate(cls, val_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.validate(val_loader, model, criterion, writer, epoch, **kwargs)

    @classmethod
    def _test(cls, test_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.test(test_loader, model, criterion, writer, epoch, **kwargs)
