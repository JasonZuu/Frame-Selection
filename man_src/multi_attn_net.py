import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchm

from .loss import RegionalIndenpendenceLoss, AGDA
from .module import Xception, Texture_Enhance, AttentionMap, AttentionPooling


class MultiAttnNet(nn.Module):
    def __init__(self,
                 num_classes=2,
                 feature_layer='b2',
                 attention_layer='final',
                 agda_mode:str=None, # agda使用的模式
                 M=8,
                 mid_dims=256,
                 dropout_rate=0.5,
                 alpha=0.05,
                 size=(224, 224),
                 margin=1,
                 inner_margin=[0.01, 0.02]):
        super().__init__()
        self.num_classes = num_classes
        self.M = M
        # structure
        self.attn_pooling = AttentionPooling()
        self.net = Xception(num_classes)
        if agda_mode is not None:
            self.use_AGDA = True
            self.AGDA = AGDA(mode=agda_mode)
        else:
            self.use_AGDA = False
        # auxiliary
        self.feature_layer = feature_layer
        self.attention_layer = attention_layer
        with torch.no_grad():
            layers = self.net(torch.zeros(1, 3, size[0], size[1]))
        num_features = layers[self.feature_layer].shape[1]
        self.mid_dims = mid_dims
        self.attentions = AttentionMap(
            layers[self.attention_layer].shape[1], self.M)
        self.texture_enhance = Texture_Enhance(num_features, M)
        self.num_features = self.texture_enhance.output_features
        self.num_features_d = self.texture_enhance.output_features_d
        self.projection_local = nn.Sequential(nn.Linear(
            M*self.num_features, mid_dims), nn.Hardswish(), nn.Linear(mid_dims, mid_dims))
        self.project_final = nn.Linear(layers['final'].shape[1], mid_dims)
        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(
            mid_dims*2, mid_dims), nn.Hardswish(), nn.Linear(mid_dims, num_classes))
        self.RI_loss = RegionalIndenpendenceLoss(
            M, self.num_features_d, num_classes, alpha, margin, inner_margin)
        self.dropout = nn.Dropout2d(dropout_rate, inplace=True)
        self.dropout_final = nn.Dropout(dropout_rate, inplace=True)
        # self.center_loss=Center_Loss(self.num_features*M,num_classes)

    def train_forward(self, x, drop_final=False):
        layers = self.net(x)
        feature_maps = layers[self.feature_layer]
        raw_attentions = layers[self.attention_layer]
        attention_maps_ = self.attentions(raw_attentions)
        dropout_mask = self.dropout(torch.ones(
            [attention_maps_.shape[0], self.M, 1], device=x.device))
        attention_maps = attention_maps_*torch.unsqueeze(dropout_mask, -1)
        feature_maps, feature_maps_d = self.texture_enhance(
            feature_maps, attention_maps_)
        feature_maps_d = feature_maps_d - \
            feature_maps_d.mean(dim=[2, 3], keepdim=True)
        feature_maps_d = feature_maps_d / \
            (torch.std(feature_maps_d, dim=[2, 3], keepdim=True)+1e-8)
        feature_matrix_ = self.attn_pooling(feature_maps, attention_maps_)
        feature_matrix = feature_matrix_*dropout_mask

        B, M, N = feature_matrix.size()
        feature_matrix = feature_matrix.view(B, -1)
        feature_matrix = F.hardswish(self.projection_local(feature_matrix))
        final = layers['final']
        attention_maps = attention_maps.sum(dim=1, keepdim=True)
        if drop_final:
            projected_final *= 0
        else:
            final = self.attn_pooling(final, attention_maps, norm=1).squeeze(1)
            final = self.dropout_final(final)
            projected_final = F.hardswish(self.project_final(final))
        feature_matrix = torch.cat((feature_matrix, projected_final), 1)
        ensemble_logit = self.ensemble_classifier_fc(feature_matrix)
        return ensemble_logit, feature_maps_d, attention_maps

    def forward(self, X):
        ensemble_logit, feature_maps_d, attention_maps = self.train_forward(X)
        return ensemble_logit

    def loss(self,
             X,
             labels,
             use_RIL=True):
        logits, feature_maps_d, attention_maps = self.train_forward(X)
        if use_RIL:
            RI_loss, feature_matrix_d = self.RI_loss(
                feature_maps_d, attention_maps, labels)
        else:
            RI_loss = 0
        if self.use_AGDA: # AGDA has bug
            agda_loss, match_loss = self.AGDA.agda(X, attention_maps)
        else:
            agda_loss, match_loss = 0, 0
        ce_loss = F.cross_entropy(logits, labels)
        return ce_loss + 0.5*RI_loss + agda_loss + 0.1*match_loss
