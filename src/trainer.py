# import sys
# sys.path.append('..')
import torch
import numpy as np
from utils.earlystopping import EarlyStopping
from utils.meter import Meter
import time
class Trainer(object):
    def __init__(self, device, tasks, data_args, model_path, 
                 n_epochs=1000, patience=20, inter_print=20):
        
        self.device = device
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.model_path = model_path
        self.metrics = data_args['metrics']
        self.norm = data_args['norm']
        if self.norm:
            self.data_mean = torch.tensor(data_args['mean']).to(self.device)
            self.data_std = torch.tensor(data_args['std']).to(self.device)
        self.loss_fn = data_args['loss_fn']
        self.n_epochs = n_epochs
        self.patience=patience
        self.inter_print = inter_print
    
    def _prepare_batch_data(self,batch_data):
        smiless, inputs, labels, masks = batch_data
        inputs.ndata['h'] = inputs.ndata['h'].to(self.device)
        labels = labels.to(self.device)
        masks = masks.to(self.device)
        return smiless, inputs, labels, masks
    
    def _train_epoch(self, model, train_loader, loss_fn, optimizer):
        model.train()
        loss_list = []
        for i, batch_data in enumerate(train_loader):
            smiless, inputs, labels, masks = self._prepare_batch_data(batch_data)
            _, predictions = model(inputs)
            if self.norm:
                labels = (labels - self.data_mean)/self.data_std
            loss = (loss_fn(predictions, labels)*(masks!=0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        return np.mean(loss_list)
    
    def _eval(self, model, data_loader):
        model.eval()
        meter = Meter(self.tasks)
        for i, batch_data in enumerate(data_loader):
            smiless, inputs, labels, masks = self._prepare_batch_data(batch_data)
            _, predictions = model(inputs)
            if self.norm:
                predictions = predictions * self.data_std + self.data_mean
            meter.update(predictions, labels, masks)
        eval_results_dict = meter.compute_metric(self.metrics)
        return eval_results_dict
    
    def _train(self, model, train_loader, val_loader, loss_fn, optimizer, stopper):
        for epoch in range(self.n_epochs):
            loss = self._train_epoch(model, train_loader, loss_fn,
                                                 optimizer)
            val_results_dict = self._eval(model, val_loader)
            early_stop = stopper.step(val_results_dict[self.metrics[0]]['mean'],
                                     model, epoch)
            if epoch % self.inter_print == 0:
                print(f"[{epoch}] training loss:{loss}")
                for metric in self.metrics:
                    print(f"val {metric}:{val_results_dict[metric]['mean']}")
            if early_stop:
                break
    
    def fit(self, model, train_loader, val_loader, test_loader):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                            model.parameters()), 
                                     lr=1e-4, weight_decay=1e-5)
        stopper = EarlyStopping(self.model_path,self.tasks, patience=self.patience)
        
        self._train(model,train_loader,val_loader,
                            self.loss_fn,optimizer,stopper)
        stopper.load_checkpoint(model)
        test_results_dict = self._eval(model, test_loader)
        for metric in self.metrics:
                    print(f"test {metric}:{test_results_dict[metric]['mean']}")
        return model, test_results_dict