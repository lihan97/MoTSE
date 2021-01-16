import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import os
from utils import load_model, makedir
from utils.metrics import cosine_similarity,spearman,pearson_matrix,pearson

class MoTSE(object):
    def __init__(self):
        pass

    def extract_knowledge(self,model_type,task,model_path,probe_data):
        
        model = load_model(model_type,1,source_model_path=model_path)
        gs = probe_data['data'][1]
        node_feats = gs.ndata['h']
        gs.ndata['h'] = Variable(node_feats,requires_grad=True)
        g_reps, predictions = model(gs)
        predictions.mean().backward()
        gradient_mul_input = gs.ndata['h'].grad * gs.ndata['h']
        local_knowledge = np.mean(gradient_mul_input.detach().numpy(),axis=-1)
        pearson_matrix_ = np.array(pearson_matrix(g_reps.detach().numpy()))[0]
        global_knowledge = pearson_matrix_[np.triu_indices(len(g_reps), k = 1)]
        np.save(lk_path, local_knowledge)
        np.save(gk_path, global_knowledge)
            
        return local_knowledge, global_knowledge
    
    def _attribution_method(self, nnodes_list, node_feats, node_feats_):
        score_list = []
        cum_sum = np.cumsum([0]+nnodes_list)
        for i in range(1,len(cum_sum)):
            nf_g = node_feats[cum_sum[i-1]:cum_sum[i]]
            nf_g_ = node_feats_[cum_sum[i-1]:cum_sum[i]]
            
            score_list.append(cosine_similarity(nf_g,nf_g_))
        return np.mean(score_list)
    
    def _mrsa(self, R, R_):
        return spearman(R, R_)
    
    def cal_sim(self,model_type,source_tasks,target_tasks,
                source_model_paths,target_model_paths,probe_data):
        # extract local and global knowledge
        s_lk,t_lk,s_gk,t_gk = [],[],[],[]
        for task,model_path in zip(source_tasks,source_model_paths):
            lk, gk = self.extract_knowledge(model_type,task,model_path,probe_data)
            s_lk.append(lk)
            s_gk.append(gk)
        for task,model_path in zip(target_tasks,target_model_paths):
            lk, gk = self.extract_knowledge(model_type,task,model_path,probe_data)
            t_lk.append(lk)
            t_gk.append(gk)
        
        #calculate and save similarity
        nnodes_list = probe_data['nnode']
        for i, target_task in enumerate(target_tasks):
            results_dict = {
                'source_task':[],
                'am': [],
                'rsa': [],
                'motse': [],
            }
            for j, source_task in enumerate(source_tasks):
                s_a = self._attribution_method(nnodes_list, t_lk[i], s_lk[j])
                s_r = self._mrsa(t_gk[i], s_gk[j])
                coef = 0.8
                s = coef * (1+s_a)/2 + (1-coef) * s_r
                results_dict['source_task'].append(source_task)
                results_dict['am'].append(s_a)
                results_dict['rsa'].append(s_r)
                results_dict['motse'].append(s)
            pd.DataFrame(results_dict).to_csv(self.path+f'{target_task}.csv',
                                             index=False,float_format='%.4f')
        print(f"Results have been saved to {self.path}.")
    def eval_source_task_recom(self,n_recoms, target_task, source_tasks,
                               scratch_result,transfer_results,similarity):
        top_ids = np.argsort(-similarity)[:n_recoms]
        motse_result = np.max(transfer_results[top_ids])
        best_result = np.max(transfer_results)
        
        print(f"{[target_task]} scrach:{scratch_result:.3f}, motse:{motse_result}, best:{best_result}")
        return scratch_result, motse_result, best_result