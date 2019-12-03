from torch import nn as nn
from torch.nn import functional as F

from models.util import upsample


class SemsegCrossEntropy(nn.Module):
    def __init__(self, num_classes=19, ignore_id=19, print_each=20):
        super(SemsegCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.ignore_id = ignore_id
        self.step_counter = 0
        self.print_each = print_each

    def loss(self, y, t):
        if y.shape[2:4] != t.shape[1:3]:
            y = upsample(y, t.shape[1:3])
        return F.cross_entropy(y, target=t, ignore_index=self.ignore_id)

    def forward(self, logits, labels, **kwargs):
        loss = self.loss(logits, labels)
        if (self.step_counter % self.print_each) == 0:
            print(f'Step: {self.step_counter} Loss: {loss.data.cpu().item():.4f}')
        self.step_counter += 1
        return loss
