# Utils
import logging
import time
import warnings

import numpy as np
# Torch related stuff
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from util.evaluation.metrics import accuracy
# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, save_image_and_log_to_tensorboard
from util.visualization.confusion_matrix_heatmap import make_heatmap


def evaluate(data_loader, model, criterion, writer, epoch, logging_label, no_cuda=False, log_interval=20, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    logging_label : string
        Label for logging purposes. Typically 'test' or 'valid'. Its prepended to the logging output path and messages.
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    -------
    top1.avg : float
        Accuracy of the model of the evaluated split
    """
    # 'Run' is injected in kwargs at runtime IFF it is a multi-run event
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_time = AverageMeter()

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)

    with torch.no_grad():
        for batch_idx, (input, target) in pbar:

            # Measure data loading time
            data_time.update(time.time() - end)

            # Moving data to GPU
            if not no_cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # Compute output
            output = model(input)

            # Compute and record the loss
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            # Compute and record the accuracy
            acc1 = accuracy(output.data, target, topk=(1,))[0]
            top1.update(acc1[0], input.size(0))

            # Get the predictions
            _ = [preds.append(item) for item in [np.argmax(item) for item in output.data.cpu().numpy()]]
            _ = [targets.append(item) for item in target.cpu().numpy()]

            # Add loss and accuracy to Tensorboard
            if multi_run is None:
                writer.add_scalar(logging_label + '/mb_loss', loss.item(), epoch * len(data_loader) + batch_idx)
                writer.add_scalar(logging_label + '/mb_accuracy', acc1.cpu().numpy(),
                                  epoch * len(data_loader) + batch_idx)
            else:
                writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.item(),
                                  epoch * len(data_loader) + batch_idx)
                writer.add_scalar(logging_label + '/mb_accuracy_{}'.format(multi_run), acc1.cpu().numpy(),
                                  epoch * len(data_loader) + batch_idx)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % log_interval == 0:
                pbar.set_description(logging_label +
                                     ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

                pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                                 Loss='{loss.avg:.4f}\t'.format(loss=losses),
                                 Acc1='{top1.avg:.3f}\t'.format(top1=top1),
                                 Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    # Make a confusion matrix
    try:
        cm = confusion_matrix(y_true=targets, y_pred=preds)
        confusion_matrix_heatmap = make_heatmap(cm, data_loader.dataset.classes)
    except ValueError:
        logging.warning('Confusion Matrix did not work as expected')

        confusion_matrix_heatmap = np.zeros((10, 10, 3))

    # Logging the epoch-wise accuracy and confusion matrix
    if multi_run is None:
        writer.add_scalar(logging_label + '/accuracy', top1.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                          image=confusion_matrix_heatmap, global_step=epoch)
    else:
        writer.add_scalar(logging_label + '/accuracy_{}'.format(multi_run), top1.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_{}'.format(multi_run),
                                          image=confusion_matrix_heatmap, global_step=epoch)

    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'Acc@1={top1.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

    # Generate a classification report for each epoch
    _log_classification_report(data_loader, epoch, preds, targets, writer)

    return top1.avg.item()


def _log_classification_report(data_loader, epoch, preds, targets, writer):
    """
    This routine computes and prints on Tensorboard TEXT a classification
    report with F1 score, Precision, Recall and similar metrics computed
    per-class.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    epoch : int
        Number of the epoch (for logging purposes)
    preds : list
        List of all predictions of the model for this epoch
    targets : list
        List of all correct labels for this epoch
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    Returns
    -------
        None
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        classification_report_string = str(classification_report(y_true=targets,
                                                                 y_pred=preds,
                                                                 target_names=[str(item) for item in
                                                                               data_loader.dataset.classes]))
    # Fix for TB writer. Its an ugly workaround to have it printed nicely in the TEXT section of TB.
    classification_report_string = classification_report_string.replace('\n ', '\n\n       ')
    classification_report_string = classification_report_string.replace('precision', '      precision', 1)
    classification_report_string = classification_report_string.replace('avg', '      avg', 1)

    writer.add_text('Classification Report for epoch {}\n'.format(epoch), '\n' + classification_report_string, epoch)
