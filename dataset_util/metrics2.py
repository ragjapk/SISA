from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, confusion_matrix
from scipy.stats import wasserstein_distance
import numpy as np
from itertools import product

def auroc_calculate(label, predict):
    return roc_auc_score(label, predict)


def auprc_calculate(label, predict):
    return average_precision_score(label, predict)


def ce_calculate(label, predict):
    def sigmoid(x):
        return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

    return log_loss(label, sigmoid(predict))


def confusion_mat_calculate(label, predict):
    tn, fp, fn, tp = confusion_matrix(label, predict).ravel()
    return [tn, fp, fn, tp]


def accuracy_calculate(confusion_mat):
    o = confusion_mat
    return (o[0] + o[3]) / (o[0] + o[1] + o[2] + o[3])


def f1_calculate(confusion_mat):
    o = confusion_mat
    return 2 * o[3] / (2 * o[3] + o[1] + o[2])


def mean_calculate(x, y):
    return (x.mean() - y.mean()) ** 2


def emd_calculate(x, y):
    return wasserstein_distance(x, y)


def fpr_gap_calculate(confusion_mat_x, confusion_mat_y):
    def fpr_calculate(confusion_mat):
        tn, fp, fn, tp = confusion_mat
        fpr = fp / (tn + fp)
        return fpr

    return np.absolute(fpr_calculate(confusion_mat_x) - fpr_calculate(confusion_mat_y))


def tpr_gap_calculate(confusion_mat_x, confusion_mat_y):
    def tpr_calculate(confusion_mat):
        tn, fp, fn, tp = confusion_mat
        tpr = tp / (tp + fn)
        return tpr

    return np.absolute(tpr_calculate(confusion_mat_x) - tpr_calculate(confusion_mat_y))


class Metrics:
    def __init__(self, groups, fairness):
        assert fairness in ['EqOdd', 'EqOpp']
        self.group_map = []
        for group in groups:
            self.group_map.append(group)
        self.fairness = fairness
    
    def separate_data_group_sisa(self,data,ad_map):
        ad = ad_map
        group_set = []
        group_num_list = []
        for i in range(len(ad)):
            if ad[i] == 1:
                group_set.append(sorted(list(set(data['group{}'.format(i+1)]))))
                group_num_list.append(i+1)
        output = dict()  
        if(len(group_num_list)>3):
            group_list = product(group_set[0], group_set[1], group_set[2], group_set[3])
            for group1, group2, group3, group4 in group_list:
                group_idx = ((data['group{}'.format(group_num_list[0])] == group1) & (data['group{}'.format(group_num_list[1])] == group2) & (data['group{}'.format(group_num_list[2])] == group3) & (data['group{}'.format(group_num_list[3])] == group4))
                soft_predict_group = data['soft_predict'][group_idx]
                hard_predict_group = data['hard_predict'][group_idx]
                label_group = data['label'][group_idx]
                o = {'soft_predict': soft_predict_group, 'hard_predict': hard_predict_group, 'label': label_group}
                output['{}{}{}{}'.format(self.group_map[0][group1],self.group_map[1][group2],self.group_map[2][group3],self.group_map[3][group4])] = o
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']}
        elif(len(group_num_list)>2):
            group_list = product(group_set[0], group_set[1], group_set[2])
            for group1, group2, group3 in group_list:
                group_idx = ((data['group{}'.format(group_num_list[0])] == group1) & (data['group{}'.format(group_num_list[1])] == group2) & (data['group{}'.format(group_num_list[2])] == group3))
                soft_predict_group = data['soft_predict'][group_idx]
                hard_predict_group = data['hard_predict'][group_idx]
                label_group = data['label'][group_idx]
                o = {'soft_predict': soft_predict_group, 'hard_predict': hard_predict_group, 'label': label_group}
                output['{}{}{}'.format(self.group_map[group_num_list[0]-1][group1],self.group_map[group_num_list[1]-1][group2],self.group_map[group_num_list[2]-1][group3])] = o
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']}
        elif(len(group_num_list)>1):
            group_list = product(group_set[0], group_set[1])
            for group1, group2 in group_list:
                group_idx = ((data['group{}'.format(group_num_list[0])] == group1) & (data['group{}'.format(group_num_list[1])] == group2))
                soft_predict_group = data['soft_predict'][group_idx]
                hard_predict_group = data['hard_predict'][group_idx]
                label_group = data['label'][group_idx]
                o = {'soft_predict': soft_predict_group, 'hard_predict': hard_predict_group, 'label': label_group}
                output['{}{}'.format(self.group_map[group_num_list[0]-1][group1],self.group_map[group_num_list[1]-1][group2])] = o
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']}
        elif(len(group_num_list)==1):
            for group in group_set[0]:
                group_idx = (data['group{}'.format(group_num_list[0])] == group)
                soft_predict_group = data['soft_predict'][group_idx]
                hard_predict_group = data['hard_predict'][group_idx]
                label_group = data['label'][group_idx]
                o = {'soft_predict': soft_predict_group, 'hard_predict': hard_predict_group, 'label': label_group}
                output['{}'.format(self.group_map[group_num_list[0]-1][group])] = o
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']}
        else:
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']}
        return output
    
    def separate_data_group(self,data,ad_map):
        ad = ad_map
        group_set = []
        group_num_list = []
        for i in range(len(ad)):
            if ad[i] == 1:
                group_set.append(sorted(list(set(data['group{}'.format(i+1)]))))
                group_num_list.append(i+1)
        output = dict()        
        if(len(group_num_list)>2):
            group_list = product(group_set[0], group_set[1], group_set[2])
            for group1, group2, group3 in group_list:
                group_idx = ((data['group{}'.format(group_num_list[0])] == group1) & (data['group{}'.format(group_num_list[1])] == group2) & (data['group{}'.format(group_num_list[2])] == group3))
                soft_predict_group = data['soft_predict'][group_idx]
                hard_predict_group = data['hard_predict'][group_idx]
                label_group = data['label'][group_idx]
                o = {'soft_predict': soft_predict_group, 'hard_predict': hard_predict_group, 'label': label_group}
                output['{}{}{}'.format(self.group_map[0][group1],self.group_map[1][group2],self.group_map[2][group3])] = o
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']}
        elif(len(group_num_list)>1):
            group_list = product(group_set[0], group_set[1])
            for group1, group2 in group_list:
                group_idx = ((data['group{}'.format(group_num_list[0])] == group1) & (data['group{}'.format(group_num_list[1])] == group2))
                soft_predict_group = data['soft_predict'][group_idx]
                hard_predict_group = data['hard_predict'][group_idx]
                label_group = data['label'][group_idx]
                o = {'soft_predict': soft_predict_group, 'hard_predict': hard_predict_group, 'label': label_group}
                output['{}{}'.format(self.group_map[group_num_list[0]-1][group1],self.group_map[group_num_list[1]-1][group2])] = o
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']}
        elif(len(group_num_list)==1):
            for group in group_set[0]:
                group_idx = (data['group{}'.format(group_num_list[0])] == group)
                soft_predict_group = data['soft_predict'][group_idx]
                hard_predict_group = data['hard_predict'][group_idx]
                label_group = data['label'][group_idx]
                o = {'soft_predict': soft_predict_group, 'hard_predict': hard_predict_group, 'label': label_group}
                output['{}'.format(self.group_map[group_num_list[0]-1][group])] = o
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']}
        else:
            output['all'] = {'soft_predict': data['soft_predict'], 'hard_predict': data['hard_predict'], 'label': data['label']} 
        return output

    def calculate_performance(self, data):
        score = dict()
        for group, dt in data.items():
            auroc = auroc_calculate(dt['label'], dt['soft_predict'])
            auprc = auprc_calculate(dt['label'], dt['soft_predict'])
            ce = ce_calculate(dt['label'], dt['soft_predict'])
            confusion_mat = confusion_mat_calculate(dt['label'], dt['hard_predict'])
            acc = accuracy_calculate(confusion_mat)
            f1 = f1_calculate(confusion_mat)
            score[group] = [np.around(auroc, 6), np.around(auprc, 6), np.around(ce, 6), np.around(acc, 6),
                            np.around(f1, 6)]
        return score

    def calculate_fairness(self, data, ad_map=None):
        if ad_map == [0,0,0,0] or ad_map == [0,0]:
            score = dict()
        else:
            score = dict()
            confusion_mat_list = []
            keys_list = [k for k,v in data.items() if k not in ['all']]
            for keys in keys_list:
                    confusion_mat = confusion_mat_calculate(data[keys]['label'], data[keys]['hard_predict'])
                    confusion_mat_list.append(confusion_mat)
            confusion_mat_list_paired = [(a, b) for idx, a in enumerate(confusion_mat_list) for b in confusion_mat_list[idx + 1:]]

            if self.fairness == 'EqOpp':
                fpr_gap = 0
                for mat in confusion_mat_list_paired:
                    fpr_gap = fpr_gap + fpr_gap_calculate(mat[0], mat[1])  
                mean = 0.0
                emd = 0.0
                keys_paired = [(a, b) for idx, a in enumerate(keys_list) for b in keys_list[idx + 1:]]
                for keys in keys_paired:
                    mean = mean + mean_calculate(data[keys[0]]['soft_predict'][data[keys[0]]['label'] == 0], data[keys[1]]['soft_predict'][data[keys[1]]['label'] == 0])
                    emd = emd + emd_calculate(data[keys[0]]['soft_predict'][data[keys[0]]['label'] == 0], data[keys[1]]['soft_predict'][data[keys[1]]['label'] == 0])
                mean = mean/len(keys_paired)
                emd = emd/len(keys_paired)        
                score['all'] = [np.around(fpr_gap, 6), np.around(mean, 6), np.around(emd, 6)]
            else:                 
                fpr_gap = 0
                tpr_gap = 0
                for mat in confusion_mat_list_paired:
                    fpr_gap = fpr_gap + fpr_gap_calculate(mat[0], mat[1])
                    tpr_gap = tpr_gap + tpr_gap_calculate(mat[0], mat[1])
                fpr_gap = (fpr_gap + tpr_gap) / 2
                mean_0 = 0.0
                mean_1 = 0.0
                emd_0 = 0.0
                emd_1 = 0.0
                keys_paired = [(a, b) for idx, a in enumerate(keys_list) for b in keys_list[idx + 1:]]
                for keys in keys_paired:
                    mean_0 = mean_0 + mean_calculate(data[keys[0]]['soft_predict'][data[keys[0]]['label'] == 0], data[keys[1]]['soft_predict'][data[keys[1]]['label'] == 0])
                    mean_1 = mean_1 + mean_calculate(data[keys[0]]['soft_predict'][data[keys[0]]['label'] == 1], data[keys[1]]['soft_predict'][data[keys[1]]['label'] == 1])
                    emd_0 = emd_0 + emd_calculate(data[keys[0]]['soft_predict'][data[keys[0]]['label'] == 0], data[keys[1]]['soft_predict'][data[keys[1]]['label'] == 0])
                    emd_1 = emd_0 + emd_calculate(data[keys[0]]['soft_predict'][data[keys[0]]['label'] == 1], data[keys[1]]['soft_predict'][data[keys[1]]['label'] == 1])

                mean = (mean_0 + mean_1) / 2                
                emd = (emd_0 + emd_1) / 2
                mean = mean/len(keys_paired)
                emd = emd/len(keys_paired)  
                score['all'] = [np.around(fpr_gap, 6), np.around(mean, 6), np.around(emd, 6)]
        return score

    def calculate_metrics(self, data, mode, ad_map=None):
        data = self.separate_data_group_sisa(data, ad_map)
        if (ad_map == [0,0,0,0]):
            performance = self.calculate_performance(data)
            fairness = None
        else:        
            performance = self.calculate_performance(data)
            fairness = self.calculate_fairness(data, ad_map)
        score = {'performance': performance, 'fairness': fairness}
        return score
