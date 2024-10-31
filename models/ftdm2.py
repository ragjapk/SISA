import torch.nn as nn
import torch.nn.functional as F
import torch
from .fairloss import MeanLoss
from torch.cuda.amp import autocast


class FATDM(nn.Module):
    def __init__(self, feature_extractor, task_classifier, domain_y_transfer, domain_ygra_transfer, alpha, model_type, n_domains, device):
        super(FATDM, self).__init__()
        self.model_type = model_type
        self.feature_extractor = feature_extractor
        self.task_classifier = task_classifier
        self.domain_y_transfer = domain_y_transfer
        self.domain_ygra_transfer = domain_ygra_transfer
        self.domain_y_transfer.eval()
        for p in self.domain_y_transfer.parameters():
            p.requires_grad = False
        self.domain_ygra_transfer.eval()
        for p in self.domain_ygra_transfer.parameters():
            p.requires_grad = False
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
                f_loss = torch.zeros(1).to(self.device)
                reg_y = c_loss.new_zeros([1])
                reg_yaa = c_loss.new_zeros([1])
                j_loss = c_loss.new_zeros([1])
                if self.training:
                    f_loss = MeanLoss(self.device, fair_criteria)
                    f_loss = f_loss(logits, y_task, sen_group_name, sen_groups, ad_map)
                        
                    loss_total = c_loss + self.alpha * (reg_yaa + reg_y) + weight * (f_loss) 
                    return loss_total, c_loss, reg_y + reg_yaa, j_loss, f_loss, logits, z
                else:
                    return c_loss, c_loss, reg_y + reg_yaa, j_loss, f_loss, logits, z
        else:
            with autocast():
                z = self.feature_extractor(feature)
                logits = self.task_classifier(z).squeeze(-1)
            return logits, z
