import torch
import torch.nn as nn
import torchvision.models as models


class FFNN(nn.Module):
    def __init__(self, in_feature, out_feature=1):
        super(FFNN, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=in_feature, out_features=64), nn.ReLU(),
                                    nn.Linear(in_features=64, out_features=32), nn.ReLU(),
                                    nn.Linear(in_features=32, out_features=out_feature))

    def forward(self, input):
        output = self.layers(input)
        return output


class LR(nn.Module):
    def __init__(self, in_feature, out_feature=1):
        super(LR, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=in_feature, out_features=out_feature))

    def forward(self, input):
        output = self.layers(input)
        return output


class DenseNet(nn.Module):
    def __init__(self, out_feature=1):
        super(DenseNet, self).__init__()
        self.layers = models.densenet121()
        self.layers.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers.classifier = nn.Linear(in_features=self.layers.classifier.in_features, out_features=out_feature,
                                           bias=True)

    def forward(self, input):
        output = self.layers(input)
        return output


class ResNet(nn.Module):
    def __init__(self, out_feature=1):
        super(ResNet, self).__init__()
        self.layers = models.resnet18()
        self.layers.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers.fc = nn.Linear(in_features=self.layers.fc.in_features, out_features=out_feature, bias=True)

    def forward(self, input):
        output = self.layers(input)
        return output


class FeatureExtractorAdult(nn.Module):
    def __init__(self, in_feature, out_feature=64):
        super(FeatureExtractorAdult, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_feature, 64), nn.ReLU(), nn.Linear(64, out_feature),
                                    nn.ReLU())

    def forward(self, input):
        features = self.layers(input)
        return features

class f(nn.Module):
    def __init__(self, n_sen, out_feature):
        super(f, self).__init__()
        self.layers = models.resnet18()
        self.layers.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.num_outputs = n_sen
        in_feature = self.layers.fc.in_features
        self.layers.fc = nn.Identity()
        self.fcs = nn.ModuleList([nn.Linear(in_features=in_feature, out_features=out_feature, bias=True) for i in range(self.num_outputs)])
        
    def forward(self, input):
        features = self.layers(input)
        outputs = [fc(features) for fc in self.fcs]
        return outputs

class ff(nn.Module):
    def __init__(self, out_feature):
        super(ff, self).__init__()
        self.layers = models.resnet18()
        self.layers.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers.fc = nn.Linear(in_features=self.layers.fc.in_features, out_features=out_feature, bias=True)

    def forward(self, input):
        features = self.layers(input)
        return features

class f1(nn.Module):
    def __init__(self, out_feature):
        super(f1, self).__init__()
        self.layers = models.resnet18()
        self.layers.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers.fc = nn.Linear(in_features=self.layers.fc.in_features, out_features=out_feature, bias=True)

    def forward(self, input):
        features = self.layers(input)
        return features

class f2(nn.Module):
    def __init__(self, out_feature):
        super(f2, self).__init__()
        self.layers = models.resnet18()
        self.layers.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers.fc = nn.Linear(in_features=self.layers.fc.in_features, out_features=out_feature, bias=True)

    def forward(self, input):
        features = self.layers(input)
        return features

class f3(nn.Module):
    def __init__(self, out_feature):
        super(f3, self).__init__()
        self.layers = models.resnet18()
        self.layers.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers.fc = nn.Linear(in_features=self.layers.fc.in_features, out_features=out_feature, bias=True)

    def forward(self, input):
        features = self.layers(input)
        return features
        
class g(nn.Module):
    def __init__(self, out_feature):
        super(g, self).__init__()
        self.layers = models.resnet18()
        self.layers.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layers.fc = nn.Linear(in_features=self.layers.fc.in_features, out_features=out_feature, bias=True)

    def forward(self, input):
        features = self.layers(input)
        return features

class h(nn.Module):
    def __init__(self, in_feature, out_feature=1):
        super(h, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=in_feature, out_features=in_feature // 2), nn.ReLU(),
                                    nn.Linear(in_features=in_feature // 2, out_features=in_feature // 4), nn.ReLU(),
                                    nn.Linear(in_features=in_feature // 4, out_features=out_feature))

    def forward(self, input):
        output = self.layers(input)
        return output


class DomainDiscriminatorG2DM(nn.Module):
    def __init__(self, in_feature, out_feature=1):
        super(DomainDiscriminatorG2DM, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_feature // 2, in_feature // 4), nn.ReLU(), nn.Dropout(),
                                    nn.Linear(in_feature // 4, out_feature))
        self.projection = nn.Linear(in_feature, in_feature // 2, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

    def forward(self, input):
        feature_proj = self.projection(input)
        domain_output = self.layers(feature_proj)
        return domain_output


class DomainDiscriminatorDANN(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(DomainDiscriminatorDANN, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_feature, in_feature // 2), nn.ReLU(), nn.Dropout(),
                                    nn.Linear(in_feature // 2, in_feature // 4),
                                    nn.ReLU(), nn.Dropout(), nn.Linear(in_feature // 4, out_feature))

    def forward(self, input):
        domain_output = self.layers(input)
        return domain_output


if __name__ == '__main__':
    f_return = f(3, 512)
    print(f_return)
    a = torch.ones([32, 2, 256, 256], dtype=torch.int32)
    op = f_return(a.float())
    print(len(op))
    print(op[0].shape)