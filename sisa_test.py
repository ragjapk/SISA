import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataset_util.datareader2 import DomainCelebDataset
from dataset_util.metrics2 import Metrics
from tqdm import tqdm
from scipy import stats
from models.networks import f, g, h
from models.stargan import Generator
from models.sisatest import SISA
from sklearn.metrics import roc_auc_score, roc_curve
from datetime import datetime
import random
import utils
import argparse
import pickle
import copy
import os
from torch.cuda.amp import GradScaler


start_time = datetime.now()

parser = argparse.ArgumentParser(description='Adult Training')
parser.add_argument('--batch_size', default=64)
parser.add_argument('--test_batch_size', default=128)
parser.add_argument('--max_epoch', default=15)
parser.add_argument('--fair_criteria', type=str, default='EqOdd')
parser.add_argument('--fair_loss', type=str, default='Mean')
parser.add_argument('--fair_weight', type=float, default=1)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--model_name', type=str, default='SISAE41')
parser.add_argument('--cov_coeff', type=float, default=0.00001)
parser.add_argument('--jacc_coeff', type=float, default=1)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--seed', default=46)
parser.add_argument('--target_dict_path', type=str, default='../data/CelebA/labels_dict')
parser.add_argument('--att_dict_path', type=str, default='../data/CelebA/att_dict')
parser.add_argument('--test_domain', type=str, default='ind_blonde')
parser.add_argument('--c_dim', type=int, default=2)
parser.add_argument('--image_feature_path', type=str, default='../data/CelebA/Img/img_npy/')

args = parser.parse_args()

SEED = int(args.seed)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

max_epoch = args.max_epoch
test_domain = args.test_domain
model_name = args.model_name
fair_criteria = args.fair_criteria
fair_loss = args.fair_loss
if fair_loss == 'None':
    fair_loss = None
fair_weight = float(args.fair_weight)
alpha = float(args.alpha)
jacc_coeff = float(args.jacc_coeff)
c_dim = args.c_dim
print(args)


domain_list = ['ind_black', 'ind_blonde', 'ind_brown']
a_list = ['attr_black', 'attr_blonde', 'attr_brown']
domain_list.remove(args.test_domain)
domain_name_alone = args.test_domain.split('_')[1]
test_att = 'attr_'+domain_name_alone
a_list.remove('attr_'+domain_name_alone)

domain_parent_path = ['../data/CelebA/']
domain_ind_list = [domain_parent_path[0]+dom for dom in domain_list]
domain_att_list = [domain_parent_path[0]+dom for dom in a_list]

test_domain_path = domain_parent_path[0]+args.test_domain
test_domain = utils.load_pkl(test_domain_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sample_weights = []
image_feature = args.image_feature_path
target_dict = utils.load_pkl(args.target_dict_path)
att_dict = utils.load_pkl(args.att_dict_path)

domain_index_list = [utils.load_pkl(dom) for dom in domain_ind_list]
domain_attribute_list = [utils.load_pkl(dom) for dom in domain_att_list]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#out_dir = 'saved_model/celeba/%s_%s_%s_%s_%s_%.4f_%.4f.ckpt' \
          #% (model_name, args.seed, args.test_domain, fair_criteria, fair_loss, fair_weight, alpha)
out_dir = 'saved_model/celeba/%s_%s_%s_%s_%s_%.4f_%.4f_%.4f_%.4f.ckpt' \
          % (model_name, args.seed, args.test_domain, fair_criteria, fair_loss, fair_weight, alpha, args.gamma, args.jacc_coeff)

group_map_b = {0: 'Not Big_Nose', 1: 'Big_Nose'}
group_map_c = {0: 'Not Smiling', 1: 'Smiling'}
group_map_m = {0: 'Not Male', 1: 'Male'}
group_map_y = {0: 'Not Young', 1: 'Young'}

metric = Metrics([group_map_b, group_map_c, group_map_m, group_map_y], args.fair_criteria)

train_dataset_list = []
val_dataset_list = []
for domain in range(len(domain_index_list)):
    train_dataset = DomainCelebDataset(domain,domain_index_list[domain], domain_attribute_list[domain], image_feature, target_dict, dataset='train',
                                            test_domain=test_domain)
    
    train_dataset_list.append(train_dataset)
    val_dataset = DomainCelebDataset(domain,domain_index_list[domain], domain_attribute_list[domain], image_feature, target_dict, dataset='val',
                                            test_domain=test_domain)
    val_dataset_list.append(val_dataset)

test_dataset = DomainCelebDataset(2, test_domain, test_att, image_feature, target_dict, dataset='test',
                                        test_domain=test_domain)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

train_dataloader_list = []
val_dataloader_list = []
sample_weights_train = []
sample_weights_val = []

for dataset in train_dataset_list:
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
    train_dataloader_list.append(dataloader)
    sample_weights_train.append(len(dataset))

for dataset in val_dataset_list:
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
    val_dataloader_list.append(dataloader)
    sample_weights_val.append(len(dataset))


feature_extractor_f = f(n_sen=4, out_feature=64)
feature_extractor_g = g(out_feature=768)
classifier = h(in_feature=1024)

domain_y_transfer = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6, img_channels=3)
domain_y_transfer.load_state_dict(torch.load('saved_model/celeba/stargan/stargan_g_%s_a_10000.ckpt'
                                             % args.test_domain, map_location=device))

domain_ygra_transfer = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6, img_channels=3)
domain_ygra_transfer.load_state_dict(torch.load('saved_model/celeba/stargan/stargan_g_%s_abcmy_10000.ckpt'
                                              % args.test_domain, map_location=device))

sen_groups = ['b', 'c', 'm', 'y']
scaler = GradScaler()

model = SISA(feature_extractor_f=feature_extractor_f, feature_extractor_g=feature_extractor_g, classifier=classifier, domain_y_transfer=domain_y_transfer,
              domain_ygra_transfer=domain_ygra_transfer, alpha=args.alpha, gamma=args.gamma, n_domains=2, device=device)

model = model.to(device)
ad_map = {'train': [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]], 'test':[[1,1,1]]}
ad_map = {'train': [[0,0], [0,1], [1,0], [1,1]], 'test':[[1,1]]}
ad_map = {'train': [[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1, 1, 0], [0,1,1,1], 
[1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]], 'test':[[0,0,0,0]]}
ad_map_list = {1: [0,0,0,1], 2: [0,0,1,0], 3: [0,0,1,1], 4: [0,1,0,0], 5: [0,1,0,1], 6: [0,1, 1, 0], 7:[0,1,1,1], 
8: [1,0,0,0], 9: [1,0,0,1], 10: [1,0,1,0], 11: [1,0,1,1], 12: [1,1,0,0], 13: [1,1,0,1], 14: [1,1, 1, 0], 15:[1,1,1,1]}
ad_map_list = {1: [0,0,0,1]}
# inference
for key in ad_map_list:
    checkpoint = torch.load(out_dir, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    model.domain_y_transfer.train()
    model.domain_ygra_transfer.train()

    with torch.no_grad():
        predict_list = np.empty(0)
        lb_list = np.empty(0)
        group_list1 = np.empty(0)
        group_list2 = np.empty(0)
        group_list3 = np.empty(0)
        group_list4 = np.empty(0)
        for i, batch in enumerate(test_dataloader):
            img = batch['img'].to(device)
            target = batch['target'].to(device)
            d_lb = batch['d_lb'].to(device)
            lb = target[:,2].to(device)
            b = target[:,7].to(device)
            c = target[:,31].to(device)
            m = target[:,20].to(device)
            y = target[:,39].to(device)
            predict, z_f, z_g, z= model(img, lb, None, c_dim, ad_map_list[key], sen_groups, [b,c,m,y], fair_criteria, fair_loss, fair_weight, jacc_coeff, 'test')
            predict_list = np.concatenate((predict_list, predict.squeeze(-1).cpu().numpy()), axis=0)
            lb_list = np.concatenate((lb_list, lb.cpu().numpy()), axis=0)
            group_list1 = np.concatenate((group_list1, b.cpu().numpy()), axis=0)
            group_list2 = np.concatenate((group_list2, c.cpu().numpy()), axis=0)
            group_list3 = np.concatenate((group_list3, m.cpu().numpy()), axis=0)
            group_list4 = np.concatenate((group_list4, y.cpu().numpy()), axis=0)
        fpr, tpr, thresholds = roc_curve(lb_list, predict_list)
        t = thresholds[np.argmax(tpr - fpr)]
        predict_list_binary = (predict_list > t) * 1
        output = {'soft_predict': predict_list, 'label': lb_list, 'group1': group_list1, 'group2': group_list2, 'group3': group_list3, 'group4': group_list4,   
                    'hard_predict': predict_list_binary}
    score = metric.calculate_metrics(output, 'test', ad_map_list[key])
    if ad_map_list[key] == [0,0,0,0]:
        print('AUROC: %.6f - AUPRC: %.6f - CE: %.6f - Acc: %.6f - F1: %.6f' % tuple(score['performance']['all']))
    else:
        print('AUROC: %.6f - AUPRC: %.6f - CE: %.6f - Acc: %.6f - F1: %.6f' % tuple(score['performance']['all']))
        print('F(T)PR gap: %.6f - Mean: %.6f - EMD: %.6f' % tuple(score['fairness']['all']))

print(stats.pearsonr(predict_list,group_list1))
print(stats.pearsonr(predict_list,group_list2))
print(stats.pearsonr(predict_list,group_list3))
print(stats.pearsonr(predict_list,group_list4))
            