import torch, torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .basic_layers import Transformer, CrossTransformer, ModalityReliabilityEstimator

class LanguageFocusedAttractor(nn.Module):
    def __init__(self, args ,dropout=0.1):
        super().__init__()

        self.lang_dim = args['model']['feature_extractor']['hidden_dims'][0]
        self.vis_dim = args['model']['feature_extractor']['hidden_dims'][1]
        self.aud_dim = args['model']['feature_extractor']['hidden_dims'][2]
        self.seq_len = args['model']['feature_extractor']['token_length'][0]
        self.heads = args['model']['feature_extractor']['heads']* 2
        self.depth = args['model']['feature_extractor']['depth'] + 1
        self.mlp_dim = self.lang_dim * 4


        self.dropout_rate = args['model']['feature_extractor'].get('dropout', 0.1)


        self.cross_v2l = CrossTransformer(
            source_num_frames=self.seq_len,
            tgt_num_frames=self.seq_len,
            dim=self.lang_dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_rate

        )
        self.cross_a2l = CrossTransformer(
            source_num_frames=self.seq_len,
            tgt_num_frames=self.seq_len,
            dim=self.lang_dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_rate

        )

        self.self_attn_refine = Transformer(
            num_frames=self.seq_len,
            save_hidden=False,
            token_len=None,
            dim=self.lang_dim,
            depth=1,
            heads=self.heads // 2,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_rate
        )


        self.norm_v = nn.LayerNorm(self.lang_dim)
        self.norm_a = nn.LayerNorm(self.lang_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.fuse_attn = nn.MultiheadAttention(
            embed_dim=self.lang_dim,
            num_heads=self.heads // 2,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.fuse_norm = nn.LayerNorm(self.lang_dim)


    def forward(self, h_v, h_a, h_l):


        h_l_v = self.cross_v2l(h_v, h_l)
        h_l_v = h_l_v[:, 1:, :]
        h_l_v = self.dropout(self.norm_v(h_l_v + h_l))


        h_l_a = self.cross_a2l(h_a, h_l)
        h_l_a = h_l_a[:, 1:, :]
        h_l_a = self.dropout(self.norm_a(h_l_a + h_l))


        fuse_input = torch.cat([h_l_v, h_l_a], dim=1)

        attn_output, _ = self.fuse_attn(query=h_l, key=fuse_input, value=fuse_input)

        attn_output = attn_output[:, :8, :]
        fused_feature = self.fuse_norm(h_l + attn_output)

        fused_feature = self.self_attn_refine(fused_feature)

        return fused_feature


