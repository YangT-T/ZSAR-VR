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

    def forward(self, query, key_value, bias=None):
        if bias is not None:
            query = query + bias.expand_as(query)
        attn_output, _ = self.attn(query, key_value, key_value)
        return attn_output

class Vrdual(Base):
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
        vr_parts = 2
        self.total_parts = total_parts
        self.vr_parts = vr_parts
        encode_input_size = self.input_size * total_parts
        encode_output_size = self.emb_dim * total_parts

        self.tmps = nn.Parameter(torch.ones([total_parts+vr_parts], dtype=torch.float) * np.log(1 / 0.07))
        if self.use_d:
            self.d = nn.Parameter(torch.ones(total_parts+vr_parts), requires_grad=True)

        self.attention = MultiHeadSelfAttention(self.input_size, self.emb_dim, k_dim, n_head)
        self.skel_encoder = nn.Sequential(
            nn.LayerNorm(encode_input_size),
            nn.Dropout(.5),
            nn.Linear(encode_input_size, self.proj_hidden_size),
            nn.SiLU(),
            HiddenLayer(self.n_hidden_layers, self.proj_hidden_size),
            nn.LayerNorm(self.proj_hidden_size),
            nn.Dropout(.5),
            nn.Linear(self.proj_hidden_size, encode_output_size),
            nn.Tanh()
        )

        self.vr_encoder = nn.Sequential(
            nn.LayerNorm(vr_parts*self.input_size),
            nn.Dropout(.5),
            nn.Linear(vr_parts*self.input_size, self.proj_hidden_size),
            nn.SiLU(),
            HiddenLayer(self.n_hidden_layers, self.proj_hidden_size),
            nn.LayerNorm(self.proj_hidden_size),
            nn.Dropout(.5),
            nn.Linear(self.proj_hidden_size, self.emb_dim * self.vr_parts),
            nn.Tanh()
        )

        self.video_reduction_layer = nn.Linear(visual_dim, input_size)
        self.fusion_proj_2 = nn.Linear(emb_dim, input_size)
        self.vr_bias_proj = nn.Linear(emb_dim, input_size)

        self.cross_attn_s2v = CrossModalAttention(embed_dim=self.input_size, num_heads=4)
        self.cross_attn_v2s = CrossModalAttention(embed_dim=self.input_size, num_heads=4)

        # self.attention_vr = MultiHeadSelfAttention(self.input_size, self.emb_dim, k_dim, n_head)
        # self.attention_bp = MultiHeadSelfAttention(self.input_size, self.emb_dim, k_dim, n_head)
        
        self.vis_enq = nn.Parameter(torch.randn(1, 4, self.input_size))

        self.cls_tokens = [clip.tokenize(cls_labels[:, i]) for i in range(total_parts)]
        self.vr_tokens = [clip.tokenize(cls_labels[:, i]) for i in [total_parts, total_parts+1]]
    
        # self.cls_tokens=[clip.tokenize(cls_labels[:, i]) for i in [0,1,2,3,4,5,6,7,9,]]

        self.clip, _ = clip.load("ViT-B/32")

        self.skel_encoder.apply(init_weights)
        self.video_reduction_layer.apply(init_weights)
        self.fusion_proj_2.apply(init_weights)
        self.vr_bias_proj.apply(init_weights)
        self.cross_attn_v2s.apply(init_weights)
        self.cross_attn_s2v.apply(init_weights)
        # self.attention_vr.apply(init_weights)
        # self.attention_bp.apply(init_weights)
        
        
        
        # # 6
        # # --- added: pre-norm for stability before cross-attn ---
        # self.pre_ln_vis = nn.LayerNorm(self.input_size)
        # self.pre_ln_vid = nn.LayerNorm(self.input_size)

        # # --- added: a correct-size fusion projection (keep old fusion_proj_2 unchanged) ---
        # self.fusion_proj_2b = nn.Linear(2 * self.input_size, self.input_size)

        # # --- added: gating for co-attn fusion (#6 will use it) ---
        # self.gate = nn.Sequential(
        #     nn.Linear(2 * self.input_size, self.input_size),
        #     nn.Sigmoid()
        # )
        # # --- added: small coef for alignment regularizer ---
        # self.align_coef = 0.05
        
        # self.pre_ln_vis.apply(init_weights)
        # self.pre_ln_vid.apply(init_weights)
        # self.fusion_proj_2b.apply(init_weights)
        # for m in self.gate: 
        #     if isinstance(m, nn.Linear): 
        #         init_weights(m)
        

        


        for name, param in self.named_parameters():
            if "clip" in name:
                param.requires_grad_(False)

    def loss(self, vis_emb, video_token_feats, target, target2, cls_idx, curr_epoch, is_train=False, is_test=False, **params):
        batch_labels = len(vis_emb) * get_rank() + torch.arange(len(vis_emb), device=vis_emb.device)

        text_features = []
        for token in self.cls_tokens:
            curr_text_features = self.clip.encode_text(token.to(vis_emb.device)).float()
            curr_text_features = curr_text_features / curr_text_features.norm(dim=-1, keepdim=True)
            text_features.append(curr_text_features.unsqueeze(0))
        text_features = torch.cat(text_features, dim=0).permute(1, 0, 2).contiguous()
        class_text_features = text_features[cls_idx]
        bp_emb = class_text_features[target]

        vr_features = []
        for token in self.vr_tokens:
            curr_vr_features = self.clip.encode_text(token.to(vis_emb.device)).float()
            curr_vr_features = curr_vr_features / curr_vr_features.norm(dim=-1, keepdim=True)
            vr_features.append(curr_vr_features.unsqueeze(0))
        vr_features = torch.cat(vr_features, dim=0).permute(1, 0, 2).contiguous()
        vr_emb = vr_features[cls_idx][target]  # (B, 2, emb_dim)

        # bias_v2s = self.vr_bias_proj(vr_emb[:, 1, :]).unsqueeze(1)  # (B, 1, 256)
        # bias_s2v = self.vr_bias_proj(vr_emb[:, 1, :]).unsqueeze(1)  # (B, 1, 256)

        bs, t, j, d = vis_emb.shape
        vis_emb = vis_emb.view(bs, -1, d)

        video_token_feats_reduced = F.adaptive_avg_pool1d(video_token_feats.permute(0,2,1), vis_emb.shape[1]).permute(0,2,1)
        if video_token_feats_reduced.shape[-1] != vis_emb.shape[-1]:
            video_token_feats_reduced = self.video_reduction_layer(video_token_feats_reduced)

        # video_enhanced = self.cross_attn_v2s(video_token_feats_reduced, vis_emb)
        # vis_enhanced = self.cross_attn_s2v(vis_emb, video_token_feats_reduced+bias_s2v)
        
        
        # vis_enhanced = self.cross_attn_s2v(vis_emb, video_token_feats_reduced)
        
        # # 1
        # video_enhanced = self.cross_attn_v2s(video_token_feats_reduced, vis_emb)
        # vis_enhanced = self.cross_attn_s2v(vis_emb, video_token_feats_reduced)
        
        # # 2 
        # video_enhanced = self.cross_attn_v2s(video_token_feats_reduced, vis_emb)
        # # vis_enq: (B, 5, 256)
        # bias_s2v = self.vr_bias_proj(vr_emb[:, 0, :]).unsqueeze(1)  # (B, 1, 256)
        # vis_enq = self.vis_enq.expand(vis_emb.size(0), -1, -1)  # shape: (B, 5, 256)
        # # video_token_feats_reduced + bias_s2v: shape (B, L, 256)
        # vis_enhanced = self.cross_attn_s2v(vis_enq, video_token_feats_reduced + bias_s2v)  # → shape: (B, 5, 256)
        # # 若后续要 concat 到 video_enhanced（shape: B, L, 256），你可以：
        # vis_enhanced = vis_enhanced.mean(dim=1, keepdim=True).expand_as(vis_emb)  # → (B, L, 256)


        # 3
        bias_v2s = self.vr_bias_proj(vr_emb[:, 0, :]).unsqueeze(1)  # (B, 1, 256)
        vis_enhanced = self.cross_attn_s2v(vis_emb, video_token_feats_reduced)
        video_enhanced = self.cross_attn_v2s(video_token_feats_reduced, vis_emb+bias_v2s)
        
        # # 4
        # bias_v2s = self.vr_bias_proj(vr_emb[:, 0, :]).unsqueeze(1)  # (B, 1, 256)
        # vis_enhanced = self.cross_attn_s2v(vis_emb, video_token_feats_reduced)
        # video_enhanced = self.cross_attn_v2s(video_token_feats_reduced, vis_emb+bias_v2s)
        
        # # 5
        # video_enhanced = self.cross_attn_v2s(video_token_feats_reduced, vis_emb)
        # vis_enhanced = video_enhanced
        
        # # # 6
        # # 对称门控共注意力：对两路都加同一 VR bias（取 vr_emb[:,1,:]）
        # # 然后分别做 v2s 与 s2v cross-attn，最后用门控+残差融合，能稳定提升泛化
        # # （不覆盖你的 #1~#5，只是一个可切换的额外策略）
        # # 使用方式：注释掉上面的版本、启用本段三行“video_enhanced / vis_enhanced / gating”即可。
        # # bias 形状: (B, 1, input_size)
        # # 注意：这里沿用你已有 self.vr_bias_proj
        # # （若想更强，可把[:,0,:] 与 [:,1,:] 分别用于两向 bias，对应不同 VR 条件）
        # # =========================================================================
        # vis_emb = self.pre_ln_vis(vis_emb)
        # video_token_feats_reduced = self.pre_ln_vid(video_token_feats_reduced)
        # video_enhanced = self.cross_attn_v2s(video_token_feats_reduced, vis_emb)
        # vis_enhanced = self.cross_attn_s2v(vis_emb, video_token_feats_reduced)
        # bias = self.vr_bias_proj(vr_emb[:, 1, :]).unsqueeze(1)      # (B, 1, 256)
        # q_vis = vis_emb + bias                                      # (B, L, 256)
        # q_vid = video_token_feats_reduced + bias                    # (B, L, 256)
        # video_enhanced_6 = self.cross_attn_v2s(q_vid, vis_emb)      # (B, L, 256)
        # vis_enhanced_6  = self.cross_attn_s2v(q_vis, video_token_feats_reduced)  # (B, L, 256)
        # # 门控残差：让两路互相提供信息但不过度污染
        # g6 = self.gate(torch.cat([video_enhanced_6, vis_enhanced_6], dim=-1))    # (B, L, 256)
        # vis_enhanced = vis_enhanced_6 + g6 * video_enhanced_6
        # video_enhanced = video_enhanced_6 + g6 * vis_enhanced_6
        # # =========================================================================

        


        fusion_feats = torch.cat([video_enhanced, vis_enhanced], dim=-1)
        fused_emb = self.fusion_proj_2(fusion_feats)

        local_vis_emb = self.attention(fused_emb, bp_emb)
        total_vis_emb = local_vis_emb.view(bs, -1)

        output_emb = self.skel_encoder(total_vis_emb)
        output_emb = output_emb.view(bs, self.total_parts, -1)

        loss = 0.
        for i in range(output_emb.shape[1]):
            curr_output_emb = output_emb[:, i, :]
            curr_att_emb = class_text_features[target][:, i, :]
            curr_loss, _ = self.loss_func(curr_output_emb, curr_att_emb, None, self.tmps[i], batch_labels)
            loss += curr_loss * self.d[i] if self.use_d else curr_loss

        global_emb = output_emb[:, -1, :]
        acc, preds, confusion_stats = accuracy(global_emb, target, class_text_features[:, -1, :], is_test, return_confusion=True)

        return {'loss': loss}, {'acc': acc, 'preds': preds}

    def configure_optimizers(self, monitor1, monitor2, lr=1e-3):
        params = []
        params += self.skel_encoder.parameters()
        params += [self.tmps]
        if self.use_d:
            params += [self.d]

        params += self.video_reduction_layer.parameters()
        params += self.fusion_proj_2.parameters()
        params += self.vr_bias_proj.parameters()
        params += self.cross_attn_v2s.parameters()
        params += self.cross_attn_s2v.parameters()
        params += self.attention.parameters()
        params += [self.vis_enq]
        
        
        
        # # 6
        # params += self.pre_ln_vis.parameters()
        # params += self.pre_ln_vid.parameters()
        # params += self.fusion_proj_2b.parameters()
        # params += self.gate.parameters()

        
        
        
        

        optimizer = optim.Adam(params, lr=lr)
        scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25, 
                                          cooldown=3, verbose=True)
        scheduler = {'scheduler': scheduler, 'monitor': monitor1, 'mode': 'min'}
        return [optimizer], [scheduler]
