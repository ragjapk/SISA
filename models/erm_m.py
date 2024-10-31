import torch.nn as nn
import torch.nn.functional as F
import torch
from .fairloss import MeanLoss
from torch.cuda.amp import autocast


class ERM(nn.Module):
    def __init__(self, feature_extractor, task_classifier, alpha, model_type, n_domains, device):
        super(ERM, self).__init__()
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier
        self.alpha = alpha
        self.device = device
        self.n_domains = n_domains

    def feature_extractor_func(self, x):
        return self.feature_extractor(x)

    def get_representation(self, feature):
        with autocast():
            return self.feature_extractor(feature)

    def forward(self, feature, y_task, y_domain, c_dim, ad_map, sen_group_name, sen_groups, fair_criteria, fair_loss, weight, dataset):
        if dataset in ['train', 'val']:
            with autocast():
                #Should it be done for individual domains?
                z = self.feature_extractor(feature)
                logits = self.task_classifier(z).squeeze(-1)
                c_loss = F.binary_cross_entropy_with_logits(logits, y_task)
                return c_loss, logits, z
         
        else:
            with autocast():
                z = self.feature_extractor(feature)
                logits = self.task_classifier(z).squeeze(-1)
            return logits, z
