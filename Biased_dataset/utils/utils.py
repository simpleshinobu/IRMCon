import numpy as np
import torch
from torch import nn, autograd
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import math
def cosine_schedule(optimizer, epoch, args, lr_init, final_epoch = None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_init
    if final_epoch is None:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warm_epochs) / (args.epochs - args.warm_epochs)))
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warm_epochs) / (final_epoch - args.warm_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def info_nce_loss(features, batch_size, temperature, normalize = False):

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
def penalty_contra(logits, y, loss_function=torch.nn.CrossEntropyLoss(), mode='w', return_flag = "abs"):
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = loss_function(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    if return_flag == "abs":
        return grad.abs()
    else:
        return grad.pow(2).mean().sqrt()
def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))
class SimpleTransform():
    def __init__(self, image_size):
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(image_size, scale=(0.3, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
    def transform(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
    def transform_batch(self, x):
        x1 = torch.zeros_like(x)
        x2 = torch.zeros_like(x)
        size = x.size(0)
        for k in range(size):
            x1[k] = self.transform(x[k])
            x2[k] = self.transform(x[k])
        return x1, x2
class MyRotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles
        self.choose_angle = 0
    def __call__(self, x):
        angle = random.choice(self.angles)
        self.choose_angle = angle
        return TF.rotate(x, angle)
    def return_angle(self):
        return self.choose_angle
class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        return loss