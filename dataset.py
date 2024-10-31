import os
import pickle
import h5py
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
#from tensorboardX import SummaryWriter
import dataloader
import utils
class CelebaModel():
    def __init__(self, opt):
        super(CelebaModel, self).__init__()
        self.epoch = 0
        self.device = opt['device']
        self.save_path = opt['save_folder']
        self.print_freq = opt['print_freq']
        self.init_lr = opt['optimizer_setting']['lr']
        #self.log_writer = SummaryWriter(os.path.join(self.save_path, 'logfile'))
        self.set_data(opt)

    def set_data(self, opt):
        """Set up the dataloaders"""
        
        data_setting = opt['data_setting']

        # normalize according to ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        if data_setting['augment']:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        image_feature = h5py.File(data_setting['image_feature_path'], 'r')
        target_dict = utils.load_pkl(data_setting['target_dict_path'])
        train_key_list = utils.load_pkl(data_setting['train_key_list_path'])
        dev_key_list = utils.load_pkl(data_setting['dev_key_list_path'])
        test_key_list = utils.load_pkl(data_setting['test_key_list_path'])
        
        train_data = dataloader.CelebaDataset(train_key_list, image_feature, target_dict, transform_train)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=1)
        dev_data = dataloader.CelebaDataset(dev_key_list, image_feature, target_dict, transform_test)
        self.dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=opt['batch_size'], shuffle=False, num_workers=1)
        test_data = dataloader.CelebaDataset(test_key_list, image_feature, target_dict, transform_test)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt['batch_size'], shuffle=False, num_workers=1)
        
        self.dev_target = np.array([target_dict[key] for key in dev_key_list])
        self.dev_class_weight = utils.compute_class_weight(self.dev_target)
        self.test_target = np.array([target_dict[key] for key in test_key_list])
        self.test_class_weight = utils.compute_class_weight(self.test_target)
        
        # We only evaluate on a subset of attributes that have enough test data on both domain
        self.subclass_idx = utils.load_pkl(data_setting['subclass_idx_path'])

#main(model, opt)
        