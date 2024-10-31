import torch.nn as nn
import torch.nn.functional as F
import torch
class MyHingeLoss(torch.nn.Module):

    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = output - target
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss