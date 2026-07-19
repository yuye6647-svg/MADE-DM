# models/prompt_generator.py
import torch, torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from .basic_layers import Transformer, CrossTransformer   #, ModalityReliabilityEstimator

class PromptGenMissingLang(nn.Module):

    def __init__(self, args, num_prompts=8, dropout=0.1):
        super().__init__()

        self.args = args
        self.lang_dim = args['model']['feature_extractor']['hidden_dims'][0]
        self.vis_dim = args['model']['feature_extractor']['hidden_dims'][1]
        self.aud_dim = args['model']['feature_extractor']['hidden_dims'][2]
        self.seq_len = args['model']['feature_extractor']['token_length'][0]
        self.num_prompts = num_prompts


        self.dropout_rate = args['model']['feature_extractor'].get('dropout', 0.1)


        self.semantic_prompts = nn.Parameter(torch.randn(1, self.num_prompts, self.lang_dim) * 0.02)
        self.emotion_prompts = nn.Parameter(torch.randn(1, self.num_prompts, self.lang_dim) * 0.02)


        self.vis_emotion_enhance = nn.Sequential(
            nn.LayerNorm(self.vis_dim),
            nn.Linear(self.vis_dim, self.vis_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.vis_dim, self.vis_dim)
        )
        self.aud_emotion_enhance = nn.Sequential(
            nn.LayerNorm(self.aud_dim),
            nn.Linear(self.aud_dim, self.aud_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.aud_dim, self.aud_dim)
        )
        self.lang_emotion_enhance = nn.Sequential(
            nn.LayerNorm(self.lang_dim),
            nn.Linear(self.lang_dim, self.lang_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.lang_dim, self.lang_dim)
        )


        self.aud_encoder = Transformer(
            num_frames=self.seq_len,
            save_hidden=False,
            token_len=None,
            dim=self.aud_dim,
            depth=1,
            heads=2,
            mlp_dim=self.aud_dim * 2,
            dropout=self.dropout_rate
        )
        self.vis_encoder = Transformer(
            num_frames=self.seq_len,
            save_hidden=False,
            token_len=None,
            dim=self.vis_dim,
            depth=1,
            heads=2,
            mlp_dim=self.vis_dim * 2,
            dropout=self.dropout_rate
        )
        self.lang_encoder = Transformer(
            num_frames=self.seq_len, save_hidden=False, token_len=None,
            dim=self.lang_dim, depth=1, heads=2, mlp_dim=self.lang_dim * 2, dropout=self.dropout_rate
        )


        self.cross_modal_fusion = CrossTransformer(
            source_num_frames=self.seq_len,
            tgt_num_frames=self.seq_len,
            dim=self.lang_dim,
            depth=1,
            heads=4,
            mlp_dim=self.lang_dim * 2,
            dropout=self.dropout_rate
        )


        self.emotion_prior = nn.Parameter(torch.randn(1, self.num_prompts, self.lang_dim) * 0.02)

        self.nonlinear_fusion = nn.Sequential(
            nn.Linear(1, 4),
            nn.GELU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1)
        )


        self.prompt_attn = nn.MultiheadAttention(
            embed_dim=self.lang_dim,
            num_heads=2,
            dropout=self.dropout_rate,
            batch_first=True
        )


        self.emotion_guide = nn.Sequential(
            nn.Linear(self.lang_dim, self.lang_dim // 2),
            nn.ReLU(),
            nn.Linear(self.lang_dim // 2, 1)
        )


        self.cond_proj = nn.Sequential(
            nn.Linear(self.vis_dim + self.aud_dim, self.lang_dim),
            nn.GELU(),
            nn.Linear(self.lang_dim, self.lang_dim),
            nn.Dropout(self.dropout_rate)
        )



        self.lang_residual_weight = nn.Sequential(
            nn.Linear(1, 4),
            nn.GELU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )


        self.norm = nn.LayerNorm(self.lang_dim)
        self.fusion_norm = nn.LayerNorm(self.lang_dim)
        self.lang_residual_norm=nn.LayerNorm(self.lang_dim)


    def forward(self, h_a, h_v , h_l, lang_completeness=None):
        b = h_a.shape[0]


        h_v_enhanced = self.vis_emotion_enhance(h_v)
        h_a_enhanced = self.aud_emotion_enhance(h_a)
        h_l_enhanced= self.lang_emotion_enhance(h_l)


        h_a_encoded = self.aud_encoder(h_a_enhanced)
        h_v_encoded = self.vis_encoder(h_v_enhanced)
        h_l_encoded = self.lang_encoder(h_l_enhanced)

        h_aud2vis = self.cross_modal_fusion(h_a_encoded, h_v_encoded)[:, 1:, :]
        h_vis2aud = self.cross_modal_fusion(h_v_encoded, h_a_encoded)[:, 1:, :]
        cross_modal_feat = (h_aud2vis + h_vis2aud) / 2

        h_l_residual = self.lang_residual_norm(h_l_encoded)



        if lang_completeness is not None:

            lang_completeness = lang_completeness.view(b, 1)
            residual_weight = self.lang_residual_weight(lang_completeness)
            residual_weight = residual_weight * 0.2
        else:
            residual_weight = torch.tensor(0.1, device=h_l.device).unsqueeze(0).repeat(b, 1)
        residual_weight = residual_weight.unsqueeze(1)


        semantic_prompt = repeat(self.semantic_prompts, '1 n d -> b n d', b=b)
        emotion_prompt = repeat(self.emotion_prompts, '1 n d -> b n d', b=b)


        prior_prompt = repeat(self.emotion_prior, '1 n d -> b n d', b=b)
        init_prompt = self.norm(semantic_prompt + emotion_prompt + 0.3 * prior_prompt + 0.1 * residual_weight * h_l_residual)

        combined_key = cross_modal_feat + 0.1 * residual_weight * h_l_residual
        combined_value = cross_modal_feat + 0.1 * residual_weight * h_l_residual

        dynamic_prompt, _ = self.prompt_attn(
            query=init_prompt,
            key=combined_key,
            value=combined_value
        )


        dynamic_prompt = self.fusion_norm(init_prompt + dynamic_prompt)


        if lang_completeness is not None:

            lang_completeness = lang_completeness.view(b, 1, 1)

            dynamic_prompt = dynamic_prompt * (1 - lang_completeness) + init_prompt * lang_completeness

        emotion_pred = self.emotion_guide(dynamic_prompt.mean(dim=1))

        return dynamic_prompt, emotion_pred

