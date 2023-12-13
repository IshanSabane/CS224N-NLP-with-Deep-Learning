import torch
from torch import nn, Tensor
from typing import Iterable, Dict

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))



class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: nn.Module, scale: float = 20.0, similarity_fct = cos_sim):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, s1_feature_batch, s2_feature_batch):
        device = torch.device('cuda')
        scores = self.similarity_fct(s1_feature_batch, s2_feature_batch) * self.scale
        labels = torch.eye(len(s1_feature_batch)).to(device)
        loss = self.cross_entropy_loss(scores, labels)
        return loss
        # reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # embeddings_a = reps[0]
        # embeddings_b = torch.cat(reps[1:])

        # scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        # labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        # return self.cross_entropy_loss(scores, labels)

    # def get_config_dict(self):
    #     return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}
