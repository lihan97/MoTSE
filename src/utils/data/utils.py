from rdkit import Chem
import dgl
from dgl.data.chem import BaseAtomFeaturizer,ConcatFeaturizer,atom_type_one_hot
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from functools import partial

def collate_molgraphs(data):
    assert len(data[0]) in [2,3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    elif len(data[0]) == 4:
        smiles, graphs, labels, masks = map(list, zip(*data))
    elif len(data[0]) == 2:
        smiles, graphs = map(list, zip(*data))
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        return smiles, bg
        
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks

def load_data(dataset,data_path,tasks,device, load=True):
    data_args = dict()
    if dataset == 'QM9':
        from utils.data.qm9 import QM9Dataset
        train_set = QM9Dataset(data_path,'train',tasks,load=load)
        val_set = QM9Dataset(data_path,'valid',tasks,load=load)
        test_set = QM9Dataset(data_path,'test',tasks,load=load)
    elif dataset == 'PCBA':
        from utils.data.pcba import PCBADataset
        train_set = PCBADataset(data_path,'train',tasks,load=load)
        val_set = PCBADataset(data_path,'valid',tasks,load=load)
        test_set = PCBADataset(data_path,'test',tasks,load=load)
    
    data_args['metrics'] = train_set.metrics
    if train_set.task_type == 'classification':
        task_pos_weights = train_set.task_pos_weights
        data_args['norm'] = None
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=task_pos_weights.to(device), 
            reduction='none')
    else:
        data_args['task_pos_weights'] = None
        data_args['norm'] = train_set.norm
        if data_args['norm']:
            data_args['mean'] = train_set.mean
            data_args['std'] = train_set.std
        loss_fn = nn.MSELoss(reduction='none')
    data_args['loss_fn'] = loss_fn
    batch_size = train_set.batch_size
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_molgraphs, 
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_molgraphs,
                            num_workers=0)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                             collate_fn=collate_molgraphs, shuffle=False,
                             num_workers=0)
    return train_loader, val_loader, test_loader, data_args

def load_probe_data(data_path, load=True):
    from utils.data.probe_data import ProbeDataset
    probe_data = ProbeDataset(data_path,load=load)
    probe_data = DataLoader(dataset=probe_data, batch_size=probe_data.batch_size,
                              shuffle=False, collate_fn=collate_molgraphs, 
                              drop_last=False, num_workers=0)
    nnode_list = np.load(data_path+'n_nodes.npy')
    probe_data = next(iter(probe_data))
    probe_data = {
        'data':probe_data,
        'name':data_path.split('/')[-2],
        'nnode':nnode_list
    }
    return probe_data

def get_node_featurizer():
        return BaseAtomFeaturizer(
        featurizer_funcs={'h': ConcatFeaturizer([
            partial(atom_type_one_hot, allowable_set=[
                'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 
                'Br', 'Te', 'I', 'At'],
                    encode_unknown=True),
        ],
        )}
    )