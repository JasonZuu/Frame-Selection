import torch
import torch.nn as nn
import torch.nn.functional as F

from man_src.module import AttentionPooling


class RegionalIndenpendenceLoss(nn.Module):
    def __init__(self,M,N,C,alpha=0.05,margin=1,inner_margin=[0.1,5]):
        super().__init__()
        self.register_buffer('feature_centers',torch.zeros(M,N))
        self.register_buffer('alpha',torch.tensor(alpha))
        self.num_classes=C
        self.margin=margin
        self.atp=AttentionPooling()
        self.register_buffer('inner_margin',torch.Tensor(inner_margin))

    def forward(self,feature_map_d,attentions,y):
        B,N,H,W=feature_map_d.size()
        B,M,AH,AW=attentions.size()
        if AH!=H or AW!=W:
            attentions=F.interpolate(attentions,(H,W),mode='bilinear',align_corners=True)
        feature_matrix=self.atp(feature_map_d,attentions)
        feature_centers=self.feature_centers
        center_momentum=feature_matrix-feature_centers
        real_mask=(y==0).view(-1,1,1)
        fcts=self.alpha*torch.mean(center_momentum*real_mask,dim=0)+feature_centers
        fctsd=fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(fctsd,torch.distributed.ReduceOp.SUM)
                    fctsd/=torch.distributed.get_world_size()
                self.feature_centers=fctsd  
        inner_margin=self.inner_margin[y]
        intra_class_loss=F.relu(torch.norm(feature_matrix-fcts,dim=[1,2])*torch.sign(inner_margin)-inner_margin)
        intra_class_loss=torch.mean(intra_class_loss)
        inter_class_loss=0
        for j in range(M):
            for k in range(j+1,M):
                inter_class_loss+=F.relu(self.margin-torch.dist(fcts[j],fcts[k]),inplace=False)
        inter_class_loss=inter_class_loss/M/self.alpha
        #fmd=attentions.flatten(2)
        #diverse_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss+inter_class_loss, feature_matrix