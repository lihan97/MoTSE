from dgl.data.chem.utils import smiles_to_bigraph
from dgl.data.utils import load_graphs, save_graphs
from dgl import backend as F
import os
import os.path as osp
import pathlib
import pandas as pd
import numpy as np
import torch
from rdkit import Chem

from .utils import get_node_featurizer

class PCBADataset(object):
    def __init__(self, data_path, split, tasks, metrics=['roc_auc', 'auprc'], load=True):
        self.all_tasks = ['PCBA-1030', 'PCBA-1458', 'PCBA-1460', 'PCBA-2546', 'PCBA-2551', 'PCBA-485297', 'PCBA-485313', 'PCBA-485364', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339', 'PCBA-504444', 'PCBA-504467', 'PCBA-588342', 'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651965', 'PCBA-652104', 'PCBA-686970', 'PCBA-686978', 'PCBA-686979', 'PCBA-720504']
        self.task_type = 'classification'
        
        self.tasks = tasks
        self.data_path = data_path
        self.split = split
        self.load = load
        self.metrics = metrics
                        
        self.preprocessed = os.path.exists(osp.join(self.data_path, 
                                                    f"{self.split}.bin")) 
        self._load()
        
    def _load(self):
        self._load_data()
        self._weight_balancing()
        num = len(self.smiles_list)
        if num >= 50000:
            self.batch_size = 512
        elif 50000 > num >= 30000:
            self.batch_size = 256
        elif 30000 > num >= 20000:
            self.batch_size = 128
        elif 20000 > num >= 10000:
            self.batch_size = 64
        elif 10000 > num >= 3000:
            self.batch_size = 32
        elif 3000 > num:
            self.batch_size = 16
        else:
            raise NotImplementedError(f'batch size not defined for {num} data ')
        print(len(self.smiles_list), "loaded!")
    
    def _load_data(self):
        if self.load and self.preprocessed:
            self.data_list,label_dict=load_graphs(osp.join(self.data_path, 
                                                    f"{self.split}.bin"))
            all_label_list,all_mask_list=label_dict['labels'],label_dict['masks']
            with open(osp.join(self.data_path,f'{self.split}_smiles.txt'),'r') as f:
                smiles_ = f.readlines()
                smiles_list = [s.strip() for s in smiles_]
        else:
            print('preprocessing data ...')
            data_file = pathlib.Path(self.data_path, f"{self.split}.csv")
            all_data = pd.read_csv(
                data_file,
                usecols=['smiles'] + self.all_tasks)
            smiless = all_data['smiles'].values.tolist()
            targets = all_data[self.all_tasks]
            self.data_list,all_label_list,smiles_list,all_mask_list,length_list=[],[],[],[],[]
            for smiles, label in zip(smiless, targets.iterrows()):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    cano_smiles = Chem.MolToSmiles(mol)
                    length = F.tensor(
                        np.array(len(cano_smiles)).astype(np.int64))
                    data = smiles_to_bigraph(cano_smiles, 
                                          node_featurizer=get_node_featurizer(),                                               edge_featurizer=None)
                    
                    label = np.array(label[1].tolist())
                    mask = np.ones_like(label)
                    mask[np.isnan(label)] = 0
                    mask = F.tensor(mask.astype(np.float32))
                    label[np.isnan(label)] = 0
                    label = F.tensor(np.array(label.astype(np.float32)))
                except Exception as e:
                    print(e)
                else:
                    self.data_list.append(data)
                    all_label_list.append(label)
                    all_mask_list.append(mask)
                    smiles_list.append(cano_smiles)
                    length_list.append(length)
            all_label_list = F.stack(all_label_list, dim=0)
            all_mask_list = F.stack(all_mask_list, dim=0)
            self.length_list = torch.stack(length_list)
            save_graphs(osp.join(self.data_path, f"{self.split}.bin"), 
                        self.data_list,
                        labels={'labels': all_label_list, 
                                'masks': all_mask_list})
            with open(osp.join(self.data_path,f"{self.split}_smiles.txt"),'w') as f:
                for smiles in smiles_list:
                    f.write(smiles + '\n')
        label_list, mask_list = [], []
        for task in self.tasks:
            label_list.append(all_label_list[:, self.all_tasks.index(task)])
            mask_list.append(all_mask_list[:, self.all_tasks.index(task)])
        self.smiles_list = np.array(smiles_list)
        self.label_list = torch.stack(label_list, dim=-1)
        self.mask_list = torch.stack(mask_list, dim=-1)
        if len(self.tasks) == 1:
            remain = (self.mask_list == 1.0).squeeze(-1)
            self.label_list = self.label_list[remain]
            self.smiles_list = self.smiles_list[remain.numpy()==1]
            self.data_list = np.array(self.data_list)[remain.numpy()==1].tolist()
            self.mask_list = torch.ones_like(self.label_list)

    def _weight_balancing(self):
        num_pos = F.sum(self.label_list, dim=0)
        num_indices = F.sum(self.mask_list, dim=0)
        self._task_pos_weights = (num_indices - num_pos) / num_pos
    @property
    def task_pos_weights(self):
        return self._task_pos_weights

    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, item):
        return self.smiles_list[item], self.data_list[item], self.label_list[item], self.mask_list[item]