import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from .basic_layers import Transformer, PreNorm


class AdaptiveInfoBottleneck(nn.Module):

    def __init__(self, dim, bottleneck_ratio=0.8, dropout=0.1):
        super().__init__()
        self.bottleneck_dim = int(dim * bottleneck_ratio)

        self.compress = nn.Sequential(
            nn.Linear(dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.reconstruct = nn.Sequential(
            nn.Linear(self.bottleneck_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout)
        )

        self.gate = nn.Parameter(torch.ones(1, 1, dim) * bottleneck_ratio)

    def forward(self, x):

        x_compressed = self.compress(x)

        x_recon = self.reconstruct(x_compressed)

        gate = torch.sigmoid(self.gate / 0.1)
        x_out = gate * x + (1 - gate) * x_recon

        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        recon_loss = torch.clamp(recon_loss, 0, 0.8)

        residual_loss = F.l1_loss(x_out, x, reduction='mean') * 0.05
        total_recon_loss = recon_loss + residual_loss

        return x_out, total_recon_loss


class ModalityContributionScorer(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, lang_feat, aud_feat, vis_feat):

        lang_global = F.layer_norm(lang_feat.mean(dim=1), (lang_feat.shape[-1],))
        aud_global = F.layer_norm(aud_feat.mean(dim=1), (aud_feat.shape[-1],))
        vis_global = F.layer_norm(vis_feat.mean(dim=1), (vis_feat.shape[-1],))

        concat_feat = torch.cat([lang_global, aud_global, vis_global], dim=-1)
        contribution = self.score_mlp(concat_feat)

        contribution = torch.clamp(contribution, min=0.01, max=0.99)

        contribution = contribution / (contribution.sum(dim=1, keepdim=True) + 1e-8)

        return contribution, lang_global, aud_global, vis_global


class CoarseToFineFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args['model']['feature_extractor']['hidden_dims'][0]
        self.dropout = args['model']['feature_extractor'].get('dropout', 0.2)
        self.bottleneck_ratio = args['model'].get('bottleneck_ratio', 0.7)

        dataset_name = args['dataset']['datasetName']
        self.is_sims = dataset_name == 'sims'



        self.coarse_align_v2l = Transformer(
            num_frames=args['model']['feature_extractor']['token_length'][0],
            save_hidden=False,
            token_len=None,
            dim=self.dim,
            depth=1,
            heads=1,
            mlp_dim=self.dim * 2,
            dropout=self.dropout
        )
        self.coarse_align_a2l = Transformer(
            num_frames=args['model']['feature_extractor']['token_length'][0],
            save_hidden=False,
            token_len=None,
            dim=self.dim,
            depth=1,
            heads=1,
            mlp_dim=self.dim * 2,
            dropout=self.dropout
        )


        self.mid_bottleneck_lang = AdaptiveInfoBottleneck(self.dim, self.bottleneck_ratio, self.dropout)
        self.mid_bottleneck_aud = AdaptiveInfoBottleneck(self.dim, self.bottleneck_ratio, self.dropout)
        self.mid_bottleneck_vis = AdaptiveInfoBottleneck(self.dim, self.bottleneck_ratio, self.dropout)
        self.mid_contribution_scorer = ModalityContributionScorer(self.dim, self.dropout)


        self.fine_cross_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=2,
            dropout=self.dropout,
            batch_first=True
        )
        self.fine_norm = nn.LayerNorm(self.dim)
        self.fine_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * 2, self.dim)
        )

        self.emotion_attn = nn.Sequential(
            nn.Linear(self.dim, 1),
            nn.Softmax(dim=1)
        )


        if self.is_sims:

            self.regressor = nn.Sequential(
                nn.LayerNorm(args['model']['dmml']['regression']['input_dim']),
                nn.Dropout(self.dropout),
                nn.Linear(args['model']['dmml']['regression']['input_dim'], self.dim),
                nn.Identity(),
                nn.Linear(self.dim, args['model']['dmml']['regression']['out_dim'])
            )
        else:


            self.regressor = nn.Sequential(
                nn.LayerNorm(args['model']['dmml']['regression']['input_dim']),
                nn.Dropout(self.dropout),
                nn.Linear(args['model']['dmml']['regression']['input_dim'], self.dim),
                nn.GELU(),
                nn.Linear(self.dim, args['model']['dmml']['regression']['out_dim'])
            )




    def forward(self, lang_feat, aud_feat, vis_feat):

        fusion_loss = 0.0

        lang_feat = F.layer_norm(lang_feat, lang_feat.shape[-1:])
        aud_feat = F.layer_norm(aud_feat, aud_feat.shape[-1:])
        vis_feat = F.layer_norm(vis_feat, vis_feat.shape[-1:])

        vis_aligned = self.coarse_align_v2l(vis_feat)
        aud_aligned = self.coarse_align_a2l(aud_feat)


        coarse_fusion = (lang_feat * 0.4 + vis_aligned * 0.3 + aud_aligned * 0.3)


        lang_bottleneck, lang_recon_loss = self.mid_bottleneck_lang(lang_feat)
        aud_bottleneck, aud_recon_loss = self.mid_bottleneck_aud(aud_aligned)
        vis_bottleneck, vis_recon_loss = self.mid_bottleneck_vis(vis_aligned)

        fusion_loss += (lang_recon_loss + aud_recon_loss + vis_recon_loss) / 4.0



        contribution, lang_global, aud_global, vis_global = self.mid_contribution_scorer(
            lang_bottleneck, aud_bottleneck, vis_bottleneck
        )


        lang_w = contribution[:, 0:1].unsqueeze(1)
        aud_w = contribution[:, 1:2].unsqueeze(1)
        vis_w = contribution[:, 2:3].unsqueeze(1)

        mid_fusion = lang_w * lang_bottleneck + aud_w * aud_bottleneck + vis_w * vis_bottleneck+ 0.1 * coarse_fusion


        fine_attn_out1, _ = self.fine_cross_attn(query=mid_fusion, key=lang_feat, value=lang_feat)

        fine_attn_out2, _ = self.fine_cross_attn(query=lang_feat, key=mid_fusion, value=mid_fusion)

        fine_attn_out = (fine_attn_out1 + fine_attn_out2) / 2.0


        fine_fusion = self.fine_norm(fine_attn_out + mid_fusion)
        fine_fusion = self.fine_norm(fine_fusion + self.fine_mlp(fine_fusion))


        attn_weights = self.emotion_attn(fine_fusion).squeeze(-1)
        global_fusion = (fine_fusion * attn_weights.unsqueeze(-1)).sum(dim=1)

        output = self.regressor(global_fusion)


        if self.is_sims:
            output = output * 2.0
            output = torch.where(output < -1.0, -1.0 + torch.tanh(output + 1.0) * 0.1, output)
            output = torch.where(output > 1.0, 1.0 - torch.tanh(output - 1.0) * 0.1, output)


        if self.training:

            mi_loss = (
                              1 - F.cosine_similarity(lang_global, aud_global, dim=-1).mean() +
                              1 - F.cosine_similarity(lang_global, vis_global, dim=-1).mean()
                      ) * 0.02
            fusion_loss += mi_loss

        return {
            'sentiment_preds': output,
            'fusion_loss': fusion_loss,
            'contribution': contribution,
            'fine_fusion_feat': fine_fusion
        }
