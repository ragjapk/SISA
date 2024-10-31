import torch.nn as nn
import torch.nn.functional as F
import torch
from .hinge_loss import MyHingeLoss
from torch.cuda.amp import autocast
from .fairloss_40 import MeanLoss
import random
import numpy as np

class SISA(nn.Module):
    def __init__(self, feature_extractor_f, feature_extractor_g, classifier, domain_y_transfer, domain_ygra_transfer, alpha, gamma, n_domains, device):
        super(SISA, self).__init__()
        self.n_domains = n_domains
        self.feature_extractor_f = feature_extractor_f
        self.feature_extractor_g = feature_extractor_g
        self.task_classifier = classifier
        self.domain_y_transfer = domain_y_transfer
        self.domain_y_transfer.eval()
        for p in self.domain_y_transfer.parameters():
            p.requires_grad = False
        self.domain_ygra_transfer = domain_ygra_transfer
        self.domain_ygra_transfer.eval()
        for p in self.domain_ygra_transfer.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.hinge_loss = MyHingeLoss()

    def forward(self, x, y_task, y_domain, c_dim, a_map, sen_group_name, sen_groups, fair_criteria, fair_loss, weight, jacc_coeff, dataset):
        self.epsilon = jacc_coeff
        if dataset in ['train', 'val']:
            #Training flow
            with autocast():
                a_d_c = torch.empty(0)
                a_d = torch.empty(0)
                adi = [torch.empty(0) for i in range(len(sen_groups))]
                
                #Randomly sampling \mathbf{C}, which is a subset of \mathcal{C}
                new_list = random.sample(a_map['train'],3)
                #new_list = a_map['train']
                new_len = len(new_list)

                for doms in new_list:
                    for i in range(len(sen_groups)):
                        adi[i] = torch.cat((adi[i],torch.Tensor([doms[i] for a in y_domain])), axis=0)
                    #reshaping c to make it match x so that it can be concatenated with x via the channel dimension
                    a_d_ = torch.Tensor([doms for a in y_domain]).unsqueeze(1).unsqueeze(1)
                    a_d_c = torch.cat((a_d_c,a_d_.repeat(1, 1, x.shape[2], (x.shape[3]//4))), axis=0)
                
                #x cloned so that x can be paired with all c's in \mathbf{C} 
                x_ = x.repeat(new_len,1,1,1)
                y_task_ = y_task.repeat(new_len)
                y_domain_ = y_domain.repeat(new_len)

                for i in range(len(sen_groups)):
                    sen_groups[i] = sen_groups[i].repeat(new_len)

                #Computing the fairness representation    
                z_f = self.feature_extractor_f(torch.cat((x_,a_d_c.to(self.device)),1))
                z_fs = torch.cat(z_f, dim=1)
                #Computing the predictive performance representation 
                z_g = self.feature_extractor_g(x)
                z_g_ = z_g.repeat(new_len,1)
                
                #Getting the final prediction after concatenating the representations
                logits = self.task_classifier(torch.cat([z_fs,z_g_],1)).squeeze(-1)
                #Computing L_{ER}/Classification Loss
                c_loss = F.binary_cross_entropy_with_logits(logits, y_task_)
                f_loss = torch.zeros(1).to(self.device)
                reg_y = c_loss.new_zeros([1])
                reg_yaa = c_loss.new_zeros([1])
                j_loss = c_loss.new_zeros([1])
                reg_yaa_p = c_loss.new_zeros([1])
                reg_yaa_n = c_loss.new_zeros([1])
                
                if self.training:
                    y_domain_new_array = torch.where(torch.arange(0,self.n_domains) != y_domain_[0].cpu())
                    idx = torch.multinomial(input=torch.tensor([1.0]), num_samples=len(y_domain_), replacement=True)
                    y_domain_new = torch.stack([y_domain_new_array[0][i] for i in idx], dim=0)
                    y_domain_new = y_domain_new.to(self.device)

                    #Creating c' by randomly shuffling c
                    rand_idx = torch.randperm(y_domain_.size(0))
                    a_d_c_new = a_d_c[rand_idx]
                    adi_new = adi[:]
                    for i in range(len(sen_groups)):
                        adi_new[i] = adi[i][rand_idx]

                    #Calculating fairness/equalized odds loss (L_{EO})
                    f_loss = MeanLoss(self.device, fair_criteria)
                    f_loss = f_loss(logits, y_task_, sen_groups, adi, new_list)

                    #Computing d' for domain translation of x from d
                    y_domain_onehot = y_domain_.new_zeros([y_domain_.shape[0], c_dim])
                    y_domain_onehot.scatter_(1, y_domain_[:, None], 1)
                    y_domain_new_onehot = y_domain_.new_zeros([y_domain_.shape[0], c_dim])
                    y_domain_new_onehot.scatter_(1, y_domain_new[:, None], 1)
                    
                    x_new2 = self.domain_ygra_transfer(x_, y_domain_onehot, y_domain_new_onehot)
                    z_new = self.feature_extractor_f(torch.cat((x_new2,a_d_c_new.to(self.device)),1))

                    #Computing indices where c=c'
                    inds_p = []
                    for i in range(len(sen_groups)):
                        inds_p.append(torch.where(adi[i] == adi_new[i])[0])

                    #Computing indices where c is not equal to c'
                    inds_n = []
                    for i in range(len(sen_groups)):
                        inds_n.append(torch.where(adi[i] != adi_new[i])[0])

                    #Computing the selective domain invariant loss (L_{DF}) between the representations z_f and z_f' based on whether c=c' or c!=c'
                    for i in range(len(sen_groups)):
                        if(inds_p[i].shape[0]):
                            reg_yaa_p = reg_yaa_p + F.mse_loss(z_new[0][inds_p[i]], z_f[0][inds_p[i]])

                    hinge_loss = MyHingeLoss()
                    for i in range(len(sen_groups)):
                        if(inds_n[i].shape[0]):
                             reg_yaa_n = reg_yaa_n + hinge_loss(self.epsilon, F.mse_loss(z_new[0][inds_n[i]], z_f[0][inds_n[i]]))

                    #Computing d' for domain translation of x from d
                    y_domain_new_array = torch.where(torch.arange(0,self.n_domains) != y_domain[0].cpu())
                    idx = torch.multinomial(input=torch.tensor([1.0]), num_samples=len(y_domain), replacement=True)
                    y_domain_new = torch.stack([y_domain_new_array[0][i] for i in idx], dim=0)
                    y_domain_new = y_domain_new.to(self.device)
                    y_domain_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_onehot.scatter_(1, y_domain[:, None], 1)
                    y_domain_new_onehot = y_domain.new_zeros([y_domain.shape[0], c_dim])
                    y_domain_new_onehot.scatter_(1, y_domain_new[:, None], 1)
                    x_new = self.domain_y_transfer(x, y_domain_onehot, y_domain_new_onehot)
                    z_new = self.feature_extractor_g(x_new)
                    
                    #Computing the domain invariant loss (L_{DG}) between the representations z_g and z_g'' 
                    reg_y = F.mse_loss(z_new, z_g)       
                    
                    #Computing the final loss
                    loss_total = c_loss + self.alpha * reg_y + self.gamma * (reg_yaa_p + reg_yaa_n) + weight * (f_loss) 
                    return loss_total, c_loss, reg_y + reg_yaa_p + reg_yaa_n, j_loss, f_loss, logits, z_fs, z_g_, torch.cat([z_fs,z_g_],1), y_task_, y_domain_, sen_groups[0], sen_groups[1], sen_groups[2], sen_groups[3]
                else:
                    return c_loss, c_loss, reg_y + reg_yaa_p + reg_yaa_n, j_loss, f_loss, logits, z_fs, z_g_, torch.cat([z_fs,z_g_],1), y_task_, y_domain_, sen_groups[0], sen_groups[1], sen_groups[2], sen_groups[3]
        else:
            #Testing Flow
            with autocast():
                a_d_c = torch.empty(0)
                for doms in a_map['test']:
                    a_d_ = torch.Tensor([doms for a in y_task]).unsqueeze(1).unsqueeze(1)
                    a_d_c_ = a_d_.repeat(1, 1, x.shape[2], (x.shape[3]//4))
                    a_d_c = torch.cat((a_d_c,a_d_c_), axis=0)
                
                z_f = self.feature_extractor_f(torch.cat((x,a_d_c.to(self.device)),1)) 
                z_fs = torch.cat(z_f, dim=1) 
                z_g = self.feature_extractor_g(x)
                logits = self.task_classifier(torch.cat([z_fs,z_g],1)).squeeze(-1)
            return logits, z_fs, z_g, torch.cat([z_fs,z_g],1)


       