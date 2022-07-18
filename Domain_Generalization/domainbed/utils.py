import torch
import torch.nn.functional as F
from torch import nn,  autograd
import numpy as np


class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
        self.bias_scale = 1.0

    def forward(self, logits, targets, requires_weight=False, weight_base=0):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight * self.bias_scale + weight_base
        if requires_weight:
            return loss, loss_weight
        return loss

def penalty_contra(logits, y, loss_function=torch.nn.CrossEntropyLoss(), return_flag = "abs"):

    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = loss_function(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    if return_flag == "abs":
        return grad.abs()
    else:
        return grad.pow(2).mean().sqrt()

def info_nce_loss(features, batch_size, temperature=0.5, normalize = False):

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)
    if normalize:
        features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return logits, labels
