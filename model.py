# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        vgg = models.vgg16(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = False
        print("Desiabled ther parameters")
        #removing the last fc layer of ResNet
        #self.resnet = nn.Sequential(*list(vgg.children())[:-1])
        #reminding myself that I will use flattening for next layers
        n_inputs = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Sequential(nn.Linear(n_inputs, 256))
        self.pre_trained_model = vgg
        #self.fc1 = nn.Linear(in_features=4096,out_features=256)
        self.norm1 = nn.BatchNorm1d(num_features=256)
        self.drop = nn.Dropout(p=.5)
        self.fc2 = nn.Linear(in_features=256,out_features=3)
        self.softmax= nn.LogSoftmax(dim=1)
    def forward(self, images):
        x = self.pre_trained_model(images)
        #x = x.view(x.size(0), -1)
        #x = self.fc1(x)
        x = self.drop(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x