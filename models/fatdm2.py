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
                    y_domain_new_array = torch.where(torch.arange(0,self.n_domains) != y_domain[0].cpu())
                    idx = torch.multinomial(input=torch.tensor([1.0]), num_samples=len(y_domain), replacement=True)
                    y_domain_new = torch.stack([y_domain_new_array[0][i] for i in idx], dim=0)
                    y_domain_new = y_domain_new.to(self.device)
                        
                    y_domain_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_onehot.scatter_(1, y_domain[:, None], 1)
                    y_domain_new_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_new_onehot.scatter_(1, y_domain_new[:, None], 1)
                    x_new = self.domain_y_transfer(feature, y_domain_onehot, y_domain_new_onehot)
                    z_new = self.feature_extractor(x_new)
                    reg_y = F.mse_loss(z_new, z)
                        
                    f_loss = MeanLoss(self.device, fair_criteria)
                    f_loss = f_loss(logits, y_task, sen_group_name, sen_groups, ad_map)
                    y_domain_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_onehot.scatter_(1, y_domain[:, None], 1)
                    y_domain_new_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_new_onehot.scatter_(1, y_domain_new[:, None], 1)
                    x_new2 = self.domain_ygra_transfer(feature, y_domain_onehot, y_domain_new_onehot)
                    z_new2 = self.feature_extractor(x_new2)
                    reg_yaa = F.mse_loss(z_new2, z)
                        
                    loss_total = c_loss + self.alpha * (reg_yaa + reg_y) + weight * (f_loss) 
                    return loss_total, c_loss, reg_y + reg_yaa, j_loss, f_loss, logits, z
                else:
                    return c_loss, c_loss, reg_y + reg_yaa, j_loss, f_loss, logits, z
        else:
            with autocast():
                z = self.feature_extractor(feature)
                logits = self.task_classifier(z).squeeze(-1)
            return logits, z
