import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F

import clip
from model import Base, init_weights
from model.utils.losses import contrastive_loss, accuracy
from model.utils.attention import MultiHeadSelfAttention
from model.utils.layers import HiddenLayer
from model.utils.contrastive_learning import get_rank

import numpy as np


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    def forward(self, query, key_value):
        attn_output, _ = self.attn(query, key_value, key_value)
        return attn_output

class Purls(Base):
    def __init__(self, num_epoch, input_size=256, emb_dim=512, bp_num=4, t_num=3, 
                 proj_hidden_size=1024, n_hidden_layers=1, n_head=2, k_dim=150, 
                 local_type='full', cls_labels=None, use_d=False, visual_dim=768):
        
        init_dict = locals().copy()
        init_dict.pop('self')
        super().__init__(**init_dict)

        self.loss_func = contrastive_loss
        self.input_size = input_size
        self.emb_dim = emb_dim
        self.bp_num = bp_num
        self.t_num = t_num
        self.proj_hidden_size = proj_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.local_type = local_type
        self.use_d = use_d

        total_parts = self.bp_num + self.t_num + 1
        self.total_parts = total_parts
        encode_input_size = self.input_size * total_parts
        encode_output_size = self.emb_dim * total_parts

        self.tmps = nn.Parameter(torch.ones([total_parts], dtype=torch.float) * np.log(1 / 0.07))
        if self.use_d:
            self.d = nn.Parameter(torch.ones(total_parts), requires_grad=True)

        print('------')
        print(self.input_size)
        print(self.emb_dim)
        print(k_dim)
        print('------')
        
        
        self.bp_emb = nn.Parameter(torch.randn(self.total_parts, self.emb_dim))

        self.attention = MultiHeadSelfAttention(self.input_size, self.emb_dim, k_dim, n_head)
        self.skel_encoder = nn.Sequential(
            # nn.BatchNorm1d(encode_input_size),
            nn.LayerNorm(encode_input_size),
            nn.Dropout(.5),
            nn.Linear(encode_input_size, self.proj_hidden_size),
            nn.SiLU(),
            HiddenLayer(self.n_hidden_layers, self.proj_hidden_size),
            # nn.BatchNorm1d(self.proj_hidden_size),
            nn.LayerNorm(self.proj_hidden_size),
            nn.Dropout(.5),
            nn.Linear(self.proj_hidden_size, encode_output_size),
            nn.Tanh()
        )
        self.video_reduction_layer = nn.Linear(visual_dim,input_size)
        # self.fusion_proj = nn.Linear(visual_dim, input_size)
        self.fusion_proj_2 = nn.Linear(emb_dim, input_size)

        self.cross_attn_s2v = CrossModalAttention(embed_dim=self.input_size, num_heads=4)
        self.cross_attn_v2s = CrossModalAttention(embed_dim=self.input_size, num_heads=4)
        # self.gating_layer = nn.Sequential(
        #     nn.Linear(input_size * 2, input_size * 2),
        #     nn.ReLU(),
        #     nn.Linear(input_size * 2, input_size),
        #     nn.Sigmoid()
        # )


        self.cls_tokens = [clip.tokenize(cls_labels[:, i]) for i in range(total_parts)]
        self.clip, _ = clip.load("ViT-B/32")

        self.attention.apply(init_weights)
        self.skel_encoder.apply(init_weights)
        self.video_reduction_layer.apply(init_weights)
        # self.fusion_proj.apply(init_weights)
        self.fusion_proj_2.apply(init_weights)
        self.cross_attn_v2s.apply(init_weights)
        self.cross_attn_s2v.apply(init_weights)
        # self.gating_layer.apply(init_weights)

        for name, param in self.named_parameters():
            if "clip" in name:
                param.requires_grad_(False)

    def configure_optimizers(self, monitor1, monitor2, lr=1e-3):
        params = []
        params += self.skel_encoder.parameters()
        params += [self.tmps]
        if self.use_d:
            params += [self.d]
        
        params += self.attention.parameters()
        params += self.video_reduction_layer.parameters()
        # params += self.fusion_proj.parameters()
        params += self.fusion_proj_2.parameters()
        params += self.cross_attn_v2s.parameters()
        params += self.cross_attn_s2v.parameters()
        # params += self.gating_layer.parameters()


        params += [self.bp_emb]

        optimizer = optim.Adam(params, lr=lr)
        scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, 
                                        cooldown=3, verbose=True)
        scheduler = {'scheduler': scheduler,
                    'monitor': monitor1,
                    'mode': 'min'
                    }
        return [optimizer], [scheduler]



    def loss(self, vis_emb, video_token_feats, target, target2, cls_idx, curr_epoch, is_train=False, is_test=False, **params):
        batch_labels = len(vis_emb) * get_rank() + torch.arange(len(vis_emb), device=vis_emb.device)

        text_features = []
        # for token in self.cls_tokens:
        #     curr_text_features = self.clip.encode_text(token.to(vis_emb.device)).float()
        #     curr_text_features = curr_text_features / curr_text_features.norm(dim=-1, keepdim=True)
        #     text_features.append(curr_text_features.unsqueeze(0))
        # text_features = torch.cat(text_features, dim=0).permute(1, 0, 2).contiguous()
        # class_text_features = text_features[cls_idx]
        # bp_emb = class_text_features[target]
        
        
        for i in range(self.bp_num + self.t_num + 1):
            curr_text_features = self.clip.encode_text(self.cls_tokens[i].to(vis_emb.device)).float()
            curr_text_features = curr_text_features / curr_text_features.norm(dim=-1, keepdim=True)
            text_features.append(curr_text_features.unsqueeze(0))
        text_features = torch.cat(text_features, dim=0)
        text_features = text_features.permute(1, 0, 2).contiguous() # cls, bp, dim
        class_text_features = text_features[cls_idx]
        bp_emb = class_text_features[target]

        # print(bp_emb.shape)

        bs, t, j, d = vis_emb.shape
        # vis_emb = vis_emb.view(bs, -1, d)
        # video_token_feats_reduced = F.adaptive_avg_pool1d(video_token_feats.permute(0,2,1), vis_emb.shape[1]).permute(0,2,1)
        # if video_token_feats_reduced.shape[-1] != vis_emb.shape[-1]:
        #     video_token_feats_reduced = self.video_reduction_layer(video_token_feats_reduced)
        # fusion_feats = torch.cat([video_token_feats_reduced, vis_emb], dim=-1)  # shape: (B, 50, 512)
        # fused_emb = self.fusion_proj_2(fusion_feats)  # shape: (B, 50, 256)
        # bp_emb = self.bp_emb.unsqueeze(0).repeat(vis_emb.size(0), 1, 1)
        
        
        # local_vis_emb = self.attention(fused_emb, bp_emb)
        # total_vis_emb = local_vis_emb.view(bs, -1)

        # output_emb = self.skel_encoder(total_vis_emb)
        # output_emb = output_emb.view(bs, self.total_parts, -1)
        
        # for i in range(self.bp_num + self.t_num + 1):
            
        #     curr_text_features = self.clip.encode_text(self.cls_tokens[i].to(vis_emb.device)).float()
        #     curr_text_features = curr_text_features / curr_text_features.norm(dim=-1, keepdim=True)
        #     text_features.append(curr_text_features.unsqueeze(0))
        # text_features = torch.cat(text_features, dim=0)
        # text_features = text_features.permute(1, 0, 2).contiguous() # cls, bp, dim
        # class_text_features = text_features[cls_idx]
        # bp_emb = class_text_features[target]
        
        # get body part adaptive features
        bs, t, j, d = vis_emb.shape
        vis_emb = vis_emb.view(bs, -1, d) # bs, tp, d
        local_vis_emb = self.attention(vis_emb, bp_emb)
        local_bp_emb = bp_emb
        total_vis_emb = local_vis_emb
        
        loss = 0.
        bs, p, d = total_vis_emb.shape
        total_vis_emb = total_vis_emb.view(bs, -1).to(total_vis_emb.device)
        output_emb = self.skel_encoder(total_vis_emb)
        output_emb = output_emb.view(bs, p, -1).to(total_vis_emb.device)

        for i in range(output_emb.shape[1]):
            curr_output_emb = output_emb[:, i, :]
            # curr_att_emb = bp_emb[:, i, :]
            curr_att_emb = class_text_features[target][:, i, :]
            curr_loss, _ = self.loss_func(curr_output_emb, curr_att_emb, None, self.tmps[i], batch_labels)
            loss += curr_loss * self.d[i] if self.use_d else curr_loss
            
        

        global_emb = output_emb[:, -1, :]
        acc, preds, confusion_stats = accuracy(global_emb, target, class_text_features[:, -1, :], is_test, return_confusion=True)

        return {'loss': loss}, {'acc': acc, 'preds': preds}
