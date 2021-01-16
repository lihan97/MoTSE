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

class ProbeDataset(object):
    def __init__(self, data_path, load=True):
        
        self.data_path = data_path
        self.load = load
        self.preprocessed = os.path.exists(osp.join(self.data_path, 
                                                    f"probe_data.bin")) 
        self._load()
        
    def _load(self):
        self._load_data()
        self.batch_size = len(self.smiles_list)
        print(len(self.smiles_list), "loaded!")
    
    def _load_data(self):
        if self.load and self.preprocessed:
            self.data_list,_=load_graphs(osp.join(self.data_path, 
                                                    f"probe_data.bin"))
            with open(osp.join(self.data_path,f'probe_data_smiles.txt'),'r') as f:
                smiles_ = f.readlines()
                smiles_list = [s.strip() for s in smiles_]
            
        else:
            print('preprocessing data ...')
            with open(self.data_path+'probe_data.txt', 'r') as f:
                lines = f.readlines()
            smiless = [l.strip('\n') for l in lines]
            self.data_list,smiles_list,self.nnodes_list = [], [], []
            for smiles in smiless:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    cano_smiles = Chem.MolToSmiles(mol)
                    data = smiles_to_bigraph(cano_smiles, 
                                             node_featurizer=get_node_featurizer(),
                                             edge_featurizer=None)
                except Exception as e:
                    print(e)
                else:
                    self.data_list.append(data)
                    smiles_list.append(cano_smiles)
                    self.nnodes_list.append(data.number_of_nodes())
            save_graphs(osp.join(self.data_path, f"probe_data.bin"), self.data_list)
            with open(osp.join(self.data_path,f"probe_data_smiles.txt"),'w') as f:
                for smiles in smiles_list:
                    f.write(smiles + '\n')
            np.save(self.data_path+'n_nodes.npy', self.nnodes_list)
        self.smiles_list = np.array(smiles_list)

    def __len__(self):
        """Length of the dataset

        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.smiles_list)
    def __getitem__(self, item):
        """Get datapoint with index

        Parameters
        ----------
        item : int
            Datapoint index

        Returns
        -------
        str
            SMILES for the ith datapoint
        DGLGraph
            DGLGraph for the ith datapoint
        Tensor of dtype float32
            Labels of the datapoint for tasks
        """
        return self.smiles_list[item], self.data_list[item]