import torch
import torch.nn.functional as F
from torch import nn
from .basic_layers import Transformer, CrossTransformer
from .bert import BertTextEncoder
from einops import rearrange, repeat
from .lfa import LanguageFocusedAttractor
from .prompt_generator import PromptGenMissingLang
from .Fusion import CoarseToFineFusion

class MADEDM(nn.Module):
    def __init__(self, args):
        super(MADEDM, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert',
                                         pretrained=args['model']['feature_extractor']['bert_pretrained'])


        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0],
                      args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][0],
                        dim=args['model']['feature_extractor']['hidden_dims'][0],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0]
                        ),
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2],
                      args['model']['feature_extractor']['hidden_dims'][2]),

            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][2],
                        dim=args['model']['feature_extractor']['hidden_dims'][2],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2]
                        ),

        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1],
                      args['model']['feature_extractor']['hidden_dims'][1]),

            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1],
                        save_hidden=False,
                        token_len=args['model']['feature_extractor']['token_length'][1],
                        dim=args['model']['feature_extractor']['hidden_dims'][1],
                        depth=args['model']['feature_extractor']['depth'],
                        heads=args['model']['feature_extractor']['heads'],
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1]
                        ),

        )


        self.completeness_check = nn.ModuleList([
            Transformer(num_frames=args['model']['dmc']['completeness_check']['input_length'],
                        save_hidden=False,
                        token_len=args['model']['dmc']['completeness_check']['token_length'],
                        dim=args['model']['dmc']['completeness_check']['input_dim'],
                        depth=args['model']['dmc']['completeness_check']['depth'],
                        heads=args['model']['dmc']['completeness_check']['heads'],
                        mlp_dim=args['model']['dmc']['completeness_check']['hidden_dim']),

            nn.Sequential(
                nn.Linear(args['model']['dmc']['completeness_check']['hidden_dim'],
                          int(args['model']['dmc']['completeness_check']['hidden_dim'] / 2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(args['model']['dmc']['completeness_check']['hidden_dim'] / 2), 1),
                nn.Sigmoid()),
        ])


        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'],
                        save_hidden=False,
                        token_len=None,
                        dim=args['model']['reconstructor']['input_dim'],
                        depth=args['model']['reconstructor']['depth'],
                        heads=args['model']['reconstructor']['heads'],
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
        ])

        self.coarse_to_fine_fusion = CoarseToFineFusion(args)

        self.lfa = LanguageFocusedAttractor(
            args=args,
            dropout=0.1)

        self.prompt_gen = PromptGenMissingLang(
            args=args,
            num_prompts=8,
            dropout=0.1
        )

    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        feat_tmp = self.completeness_check[0](h_1_l)[:, :1].squeeze()
        w = self.completeness_check[1](feat_tmp)

        h_lfa = self.lfa(h_1_v, h_1_a, h_1_l)

        h_prompt, emotion_pred = self.prompt_gen(h_1_a, h_1_v, h_1_l, lang_completeness=w)

        w = w.view(b, 1, 1)

        h_lang = w * h_lfa + (1 - w) * h_prompt

        fusion_out = self.coarse_to_fine_fusion(
            lang_feat=h_lang,
            aud_feat=h_1_a,
            vis_feat=h_1_v
        )

        rec_feats, complete_feats = None, None
        if (vision is not None) and (audio is not None) and (language is not None):

            rec_feat_a = self.reconstructor[0](h_1_a)
            rec_feat_v = self.reconstructor[1](h_1_v)
            rec_feat_l = self.reconstructor[2](h_1_l)
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)


            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat],
                                       dim=1)

        return {'sentiment_preds': fusion_out['sentiment_preds'],
                'w': w.squeeze(2),
                'emotion_pred': emotion_pred,
                'rec_feats': rec_feats,
                'complete_feats': complete_feats,
                'modality_contribution': fusion_out['contribution'],
                'fusion_loss': fusion_out['fusion_loss'],
                'fine_fusion_feat': fusion_out['fine_fusion_feat']
                }

def build_model(args):
    return MADEDM(args)