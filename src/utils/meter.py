import torch
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score
from utils.metrics import auprc

class Meter(object):
    def __init__(self, tasks):
        self.tasks = tasks
        self.n_tasks = len(self.tasks)

        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())
    def auprc_score(self):
        y_pred = torch.sigmoid(self.y_pred)
        scores_dict = dict()
        scores = []
        for task_id, task in enumerate(self.tasks):
            task_w = self.mask[:, task_id]
            task_y_true = self.y_true[:, task_id][task_w != 0].numpy()
            task_y_pred = y_pred[:, task_id][task_w != 0].numpy()
            score = auprc(task_y_true, task_y_pred)
            scores_dict[task] = score
            scores.append(score)
        scores_dict['mean'] = np.mean(scores)
        return scores_dict
    def roc_auc_score(self):
        y_pred = torch.sigmoid(self.y_pred)
        scores_dict = dict()
        scores = []
        for task_id, task in enumerate(self.tasks):
            task_w = self.mask[:, task_id]
            task_y_true = self.y_true[:, task_id][task_w != 0].numpy()
            task_y_pred = y_pred[:, task_id][task_w != 0].numpy()

            score = roc_auc_score(task_y_true, task_y_pred)
            scores_dict[task] = score
            scores.append(score)
        scores_dict['mean'] = np.mean(scores)
        return scores_dict

    def r2(self):
        scores_dict = dict()
        scores = []
        for task_id, task in enumerate(self.tasks):
            task_w = self.mask[:, task_id]
            task_y_true = self.y_true[:, task_id][task_w != 0].numpy()
            task_y_pred = self.y_pred[:, task_id][task_w != 0].numpy()
            score = r2_score(task_y_true, task_y_pred)
            scores_dict[task] = score
            scores.append(score)
        scores_dict['mean'] = np.mean(scores)
        return scores_dict
    def mae(self):
        scores_dict = dict()
        scores = []
        for task_id, task in enumerate(self.tasks):
            task_w = self.mask[:, task_id]
            task_y_true = self.y_true[:, task_id][task_w != 0]
            task_y_pred = self.y_pred[:, task_id][task_w != 0]
            score = torch.mean(torch.abs(task_y_pred - task_y_true)).cpu().item()
            scores_dict[task] = score
            scores.append(score)
        scores_dict['mean'] = np.mean(scores)
        return scores_dict

    def compute_metric(self, metric_names, reduction='mean'):
        self.mask = torch.cat(self.mask, dim=0)
        self.y_pred = torch.cat(self.y_pred, dim=0)
        self.y_true = torch.cat(self.y_true, dim=0)
        results_dict = dict()
        for metric_name in metric_names:
            results_dict[metric_name] = dict()
            if metric_name == 'roc_auc':
                results_dict[metric_name] = self.roc_auc_score()
            if metric_name == 'auprc':
                results_dict[metric_name] = self.auprc_score()
            if metric_name == 'mae':
                results_dict[metric_name] = self.mae()
            if metric_name == 'r2':
                results_dict[metric_name] = self.r2()
        return results_dict