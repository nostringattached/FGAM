#!/usr/bin/python

import numpy as np
from scipy import io
import os
import time
import sys

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torchnet.meter as meter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
plt.switch_backend('agg')

# _, term_width = os.popen('stty size', 'r').read().split()
term_width = int(81)
TOTAL_BAR_LENGTH = 60
last_time = time.time()
begin_time = last_time

def progressBar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def saveCheckpoint(state, filename, is_best=False):
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'best_'+filename)

def saveMkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


class LogMeters(object):
    def __init__(self, name=None, n_classes=2):
        self.name = name
        self.n_classes = n_classes
        self.path = os.path.join('log', name)
        self.conf_mtr = meter.ConfusionMeter(n_classes)
        self.auc_mtr = meter.AUCMeter()
        self.err_mtr = meter.ClassErrorMeter(topk=[1], accuracy=True)
        saveMkdir(self.path)

        self.fp = open(os.path.join(self.path, 'res.log'), 'w')
        self.y_scores = np.array([], dtype=np.float32).reshape(0, 1)
        self.y_true = np.array([], dtype=np.float32).reshape(0, 1)

    def update(self, output, target):
        preds = output.data
        probs = torch.exp(preds)
        _, predicted = torch.max(probs, 1)
        self.conf_mtr.add(predicted, target.data)
        if self.n_classes == 2:
            self.auc_mtr.add(probs[:, 1],
                             target.data)
            curr_output = probs[:, 1].cpu().squeeze().numpy()
            curr_output.resize(curr_output.shape[0], 1)
            curr_target = target.data.cpu().squeeze().numpy()
            curr_target.resize(curr_target.shape[0], 1)
            self.y_scores = np.vstack([self.y_scores, curr_output])
            self.y_true = np.vstack([self.y_true, curr_target])

        self.err_mtr.add(probs, target.data)

    def printLog(self, epoch=0):
        conf_mtrx = self.conf_mtr.value()
        print(conf_mtrx)
        if self.n_classes == 2:
            val_auc = roc_auc_score(self.y_true, self.y_scores)
            print('\tAUC is {:.6f}'.format(val_auc))
            average_precision = average_precision_score(self.y_true,
                                                        self.y_scores)
            print('\tAPR is {:.6f}'.format(average_precision))
            precision, recall, _ = precision_recall_curve(self.y_true,
                                                          self.y_scores)
            np.savetxt(self.path+'/precision_'+str(epoch)+'.txt',
                       precision, delimiter=',')
            np.savetxt(self.path+'/recall_'+str(epoch)+'.txt',
                       recall, delimiter=',')
            np.savetxt(self.path+'/true_'+str(epoch)+'.txt',
                       self.y_true, delimiter=',')
            np.savetxt(self.path+'/pred_'+str(epoch)+'.txt',
                       self.y_scores, delimiter=',')
            fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
            np.savetxt(self.path+'/fpr_'+str(epoch)+'.txt',
                       fpr, delimiter=',')
            np.savetxt(self.path+'/tpr_'+str(epoch)+'.txt',
                       tpr, delimiter=',')
        acc = self.err_mtr.value()
        acc = acc[0]
        print('\tACC is {:.6f}'.format(acc))
        self.fp.writelines('Confusion Matrix for ' + self.name+'\n')
        self.fp.writelines(str(conf_mtrx)+'\n')
        self.fp.writelines('AUC is {:.4f}'.format(val_auc)+'\n')
        self.fp.writelines('APR is {:.4f}'.format(average_precision)+'\n')
        self.fp.writelines('ACC Rate is {:.4f}%'.format(acc)+'\n')
        self.fp.writelines('\n')
        self.fp.flush()

        # plot image
        fig = plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title('Mortality Precision-Recall curve: AP={0:0.4f}'.format(
                      average_precision))
        fig.savefig(self.path+'/precision_recall_curve_' +
                    str(epoch) + '.pdf')
        plt.close(fig)

        fig = plt.figure()
        plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')

        plt.xlabel('False Postive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim([0.0, 1.0])
        plt.xlim([0.0, 1.0])
        plt.title('Mortality ROC curve: AUC={0:0.4f}'.format(val_auc))

        fig.savefig(self.path+'/ROC_curve_' + str(epoch) + '.pdf')
        plt.close(fig)

    def reset(self):
        self.y_scores = np.array([]).reshape(0, 1)
        self.y_true = np.array([]).reshape(0, 1)
        self.conf_mtr.reset()
        self.auc_mtr.reset()
        self.err_mtr.reset()

