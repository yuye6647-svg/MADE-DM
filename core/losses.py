from torch import nn
from torch.nn import functional as F

class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.delta = args['base']['delta']
        self.epsilon = args['base']['epsilon']
        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss()

    def forward(self, out, label):
        l_cc = self.MSE_Fn(out['w'], label['completeness_labels']) if out['w'] is not None else 0

        l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out[
            'complete_feats'] is not None else 0

        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])

        l_emotion_guide = 0
        if 'emotion_pred' in out and out['emotion_pred'] is not None:
            l_emotion_guide = self.MSE_Fn(out['emotion_pred'], label['sentiment_labels'])

        l_fusion = out['fusion_loss'] if 'fusion_loss' in out else 0



        loss = self.alpha * l_cc + self.gamma * l_rec + self.sigma * l_sp + self.delta * l_emotion_guide + self.epsilon * l_fusion

        return {'loss': loss, 'l_sp': l_sp, 'l_cc': l_cc, \
                'l_rec': l_rec,'l_emotion_guide': l_emotion_guide,'l_fusion': l_fusion}
