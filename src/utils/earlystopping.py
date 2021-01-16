import torch
class EarlyStopping(object):
    def __init__(self, model_path, tasks, mode='higher', patience=80):

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
            self.best_score = -999

        else:
            self._check = self._check_lower
            self.best_score = 999
        
        self.patience = patience
        self.counter = 0
        self.model_path = model_path + '_'.join(tasks) + '.pth'
        self.best_epoch = 0
        self.early_stop = False
        self.refresh = False
    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch)
            self.best_epoch = epoch
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model, epoch)
            self.best_epoch = epoch
            self.counter = 0
            self.refresh=True
        else:
            self.counter += 1
            self.refresh=False
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model, epoch):
        torch.save({'model_state_dict': model.state_dict()}, self.model_path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.model_path)['model_state_dict'])
