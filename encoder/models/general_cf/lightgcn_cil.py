import torch as t
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, ssl_con_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
def cross_modal_align_loss(cf_emb, modal_emb):
    cf_emb = F.normalize(cf_emb, dim=-1)
    modal_emb = F.normalize(modal_emb, dim=-1)
    cos_loss = 1 - F.cosine_similarity(cf_emb, modal_emb, dim=-1).mean()
    l1_loss = F.smooth_l1_loss(cf_emb, modal_emb)
    return cos_loss + l1_loss

class ExpertLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MoEEnhancedAdapter(nn.Module):
    def __init__(self, n_exps=2, input_dim=64, output_dim=64, dropout=0.1):
        super().__init__()
        self.n_exps = n_exps
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, output_dim, dropout) for _ in range(n_exps)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, n_exps),
            nn.GELU(),
            nn.Linear(n_exps, n_exps)
        )

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        outputs = [expert(x).unsqueeze(-1) for expert in self.experts]
        outputs = t.cat(outputs, dim=-1)
        return t.sum(outputs * gate_weights.unsqueeze(1), dim=-1)

class LightGCN_cil(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_cil, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.device = configs['device']
        self.embedding_size = configs['model']['embedding_size']
        self.sim_scale = self.hyper_config['sim_scale']
        self.sem_base = self.hyper_config['sem_base']
        self.sem_ratio = self.hyper_config['sem_ratio']
        self.keep_rate = self.hyper_config['keep_rate']
        self.user_embeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.item_num, self.embedding_size)))
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().to(self.device)
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().to(self.device)
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.ssl_lambda = self.hyper_config['ssl_lambda']
        self.tau = self.hyper_config['tau']
        self.n_exps = self.hyper_config['n_exps']
        self.moe_dropout = self.hyper_config['moe_dropout']
        self.prf_moe = MoEEnhancedAdapter(
            n_exps=self.n_exps,
            input_dim=self.usrprf_embeds.shape[1],
            output_dim=self.embedding_size,
            dropout=self.moe_dropout
        )
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False


    def semantic_guided_propagate(self, adj, embeds, user_emb_llm, item_emb_llm):
        u_llm = F.normalize(user_emb_llm, dim=-1)
        i_llm = F.normalize(item_emb_llm, dim=-1)
        all_llm = t.cat([u_llm, i_llm], dim=0)

        with t.no_grad():
            adj_coalesced = adj.coalesce()
            edge_index = adj_coalesced.indices()
            row, col = edge_index[0], edge_index[1]
            sim = (all_llm[row] * all_llm[col]).sum(dim=-1)
            sim = (sim + 1.0) * self.sim_scale
            semantic_weight = self.sem_base + self.sem_ratio * sim

        guided_adj = t.sparse_coo_tensor(
            edge_index,
            adj_coalesced.values() * semantic_weight,
            size=adj_coalesced.size(),
            device=adj.device,
            dtype=adj.dtype
        )
        return t.spmm(guided_adj, embeds)

    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)

    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        embeds = t.cat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        user_llm = self.prf_moe(self.usrprf_embeds)
        item_llm = self.prf_moe(self.itmprf_embeds)
        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)
        for _ in range(self.layer_num):
            embeds = self.semantic_guided_propagate(adj, embeds_list[-1], user_llm, item_llm)
            embeds_list.append(embeds)
        embeds = t.stack(embeds_list, dim=1).mean(dim=1)
        user_emb, item_emb = embeds[:self.user_num], embeds[self.user_num:]
        self.final_embeds = t.cat([user_emb, item_emb], dim=0)
        return user_emb, item_emb

    def _pick_embeds(self, user_emb, item_emb, batch_data):
        ancs, poss, negs = batch_data
        return user_emb[ancs], item_emb[poss], item_emb[negs]

    def cal_loss(self, batch_data):
        self.is_training = True
        ancs, poss, negs = batch_data
        user_emb, item_emb = self.forward(self.adj, self.keep_rate)
        anc_emb, pos_emb, neg_emb = self._pick_embeds(user_emb, item_emb, batch_data)
        with t.no_grad():
            usrprf = self.usrprf_embeds
            itmprf = self.itmprf_embeds

        moe_user = self.prf_moe(usrprf)
        moe_item = self.prf_moe(itmprf)
        anc_llm, pos_llm, _ = self._pick_embeds(moe_user, moe_item, batch_data)
        bpr_loss = cal_bpr_loss(anc_emb, pos_emb, neg_emb) / ancs.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        cl_loss = (cross_modal_align_loss(anc_emb, anc_llm) + cross_modal_align_loss(pos_emb, pos_llm)) * self.kd_weight
        na_loss = (self.get_neighbor_aggregate_loss(anc_emb, pos_emb, self.tau) + self.get_neighbor_aggregate_loss(anc_llm, pos_llm, self.tau)) * self.ssl_lambda
        total_loss = bpr_loss + reg_loss + cl_loss + na_loss
        losses = {
            'bpr': bpr_loss.item(),
            'reg': reg_loss.item(),
            'cl': cl_loss.item(),
            'na': na_loss.item(),
            'total': total_loss.item()
        }
        return total_loss.clamp(min=-10, max=1e4), losses

    @staticmethod
    def get_neighbor_aggregate_loss(emb1, emb2, tau):
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)

        pos_score = t.exp((emb1 * emb2).sum(dim=-1) / tau)
        total_score = t.exp((t.matmul(emb1, emb2.t()) + t.matmul(emb1, emb1.t())) / tau).sum(dim=-1)

        return -t.log(pos_score / (total_score + 1e-8) + 1e-8).mean()

    def full_predict(self, batch_data):
        self.is_training = False
        with t.no_grad():
            user_emb, item_emb = self.forward(self.adj, keep_rate=1.0)
            pck_users, train_mask = batch_data
            full_preds = user_emb[pck_users.long()] @ item_emb.t()
        return self._mask_predict(full_preds, train_mask)