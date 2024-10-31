import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .data_utils import read_metadata, convert_age_2_num, get_age_range, convert_gender_2_num, convert_race_2_num
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
import torch

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
transform_test = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,])

class ConcatDomainCelebDataset(Dataset):
    def __init__(self, domain_index_list, domain_attribute_list, image_feature, target_dict, att_dict, dataset, test_domain, lb_val, b_val, c_val, m_val, y_val):
        assert dataset in ['train', 'val', 'test']
        domain_indices = {}
        domain_attributes = {}
        self.transform = transform_train
        self.target_dict= target_dict
        self.image_feature = image_feature
        self.domains = []
        for d in range(len(domain_index_list)):
            if lb_val is not None and b_val is not None and c_val is not None and m_val is not None and y_val is not None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Big_Nose'] == b_val) & (domain_attribute_list[d]['Smiling'] == c_val) & (domain_attribute_list[d]['Male'] == m_val) & (domain_attribute_list[d]['Young'] == y_val)]
            elif lb_val is not None and b_val is not None and c_val is not None and m_val is not None and y_val is None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Big_Nose'] == b_val) & (domain_attribute_list[d]['Smiling'] == c_val) & (domain_attribute_list[d]['Male'] == m_val)]
            elif lb_val is not None and b_val is not None and c_val is not None and m_val is None and y_val is not None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Big_Nose'] == b_val) & (domain_attribute_list[d]['Smiling'] == c_val) & (domain_attribute_list[d]['Young'] == y_val)]
            elif lb_val is not None and b_val is not None and c_val is None and m_val is not None and y_val is not None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Big_Nose'] == b_val) & (domain_attribute_list[d]['Male'] == m_val) & (domain_attribute_list[d]['Young'] == y_val)]
            elif lb_val is not None and b_val is None and c_val is not None and m_val is not None and y_val is not None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Smiling'] == c_val) & (domain_attribute_list[d]['Male'] == m_val) & (domain_attribute_list[d]['Young'] == y_val)]
            elif lb_val is not None and b_val is not None and c_val is not None and m_val is None and y_val is None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Big_Nose'] == b_val) & (domain_attribute_list[d]['Smiling'] == c_val)]
            elif lb_val is not None and b_val is None and c_val is None and m_val is not None and y_val is not None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Male'] == m_val) & (domain_attribute_list[d]['Young'] == y_val)]
            elif lb_val is not None and b_val is not None and c_val is None and m_val is None and y_val is not None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Big_Nose'] == b_val) & (domain_attribute_list[d]['Young'] == y_val)]
            elif lb_val is not None and b_val is None and c_val is not None and m_val is not None and y_val is None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Smiling'] == c_val) & (domain_attribute_list[d]['Male'] == m_val)]
            elif lb_val is not None and b_val is not None and c_val is None and m_val is not None and y_val is None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Big_Nose'] == b_val) & (domain_attribute_list[d]['Male'] == m_val)]
            elif lb_val is not None and b_val is None and c_val is not None and m_val is None and y_val is not None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) &  (domain_attribute_list[d]['Smiling'] == c_val) & (domain_attribute_list[d]['Young'] == y_val)]
            elif lb_val is not None and b_val is not None and c_val is None and m_val is None and y_val is None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) &  (domain_attribute_list[d]['Big_Nose'] == b_val)]
            elif lb_val is not None and b_val is None and c_val is not None and m_val is None and y_val is None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Smiling'] == c_val)]
            elif lb_val is not None and b_val is None and c_val is None and m_val is not None and y_val is None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val)  & (domain_attribute_list[d]['Male'] == m_val)]
            elif lb_val is not None and b_val is None and c_val is None and m_val is not None and y_val is not None:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val) & (domain_attribute_list[d]['Young'] == y_val)]
            else:
                indices = domain_index_list[d][(domain_attribute_list[d]['Attractive'] == lb_val)]
            domain_indices[d] = indices
            self.domains.append(np.array([d]*len(indices)))
        #self.domains = np.array(self.domains)
        #print(self.domains)
        if dataset == 'test': #worry when you have FATDM/SISA
            self.metadata = metadata_list[3].to_numpy()
        else:
            train_metadata_list = []
            val_metadata_list = []
            train_domain_list = []
            val_domain_list = []
            for d, mtdt in domain_indices.items():
                if d != 2:
                    idx = list(range(len(mtdt)))
                    #target = self.target_dict[mtdt][:,2]
                    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
                    train_metadata_list.append(mtdt[train_idx])
                    val_metadata_list.append(mtdt[val_idx])
                    train_domain_list.extend(self.domains[d][train_idx])
                    val_domain_list.extend(self.domains[d][val_idx])
            if dataset == 'train':
                self.metadata = np.array(train_metadata_list[0])
                self.domains = train_domain_list
            else:
                self.metadata = np.array(val_metadata_list[0])
                self.domains = val_domain_list
        self.dataset = dataset

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        key = self.metadata[idx]
        domain = self.domains[idx]
        img_path = self.image_feature + '/' + key + '.npy'
        img = np.load(img_path)
        img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
        target = self.target_dict[key]
        if self.dataset in ['train', 'val']:
            return {'img': img, 'target':torch.FloatTensor(target), 'dlb': domain}
        else:
            return {'img': img, 'target':torch.FloatTensor(target)}

class DomainCelebDataset(Dataset):
    def __init__(self, dlb_val, ind_list, att_list, image_feature, target_dict, dataset, test_domain):
        assert dataset in ['train', 'val', 'test']
        self.target_dict = target_dict
        self.image_feature = image_feature
        group_map_b = {0: 'Not Big_Nose', 1: 'Big_Nose'}
        group_map_c = {0: 'Not Smiling', 1: 'Smiling'}
        group_map_m = {0: 'Not Male', 1: 'Male'}
        group_map_y = {0: 'Not Young', 1: 'Young'}

        if dataset == 'test':
            self.data = np.array(ind_list)
            self.domains = [dlb_val]*len(ind_list)
        else:
            idx = list(range(len(ind_list)))
            train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)

            if dataset == 'train':
                self.data = np.array(ind_list[train_idx])
                self.domains = [dlb_val]*len(ind_list[train_idx])
            else:
                self.data =np.array(ind_list[val_idx])
                self.domains = [dlb_val]*len(ind_list[val_idx])

        self.dataset = dataset
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d_lb = self.domains[idx]
        img_path = self.image_feature + '/' + self.data[idx] + '.npy'
        img = np.load(img_path)
        img = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
        target = self.target_dict[self.data[idx]]
        return {'img': img, 'target': target, 'd_lb': d_lb}
    

