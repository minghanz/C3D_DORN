"""
This is to implement the per-point depth classifier. 
For selected (or all) pixels, given a set of sampled depth hypothesis, output the occupancy or SDF of the depth hypothesis at this pixel. 
The training is via per-point per-depth binary cross entropy. 
The inference is via re-fit a simple logistic regression on sampled depth according to their predicted label. 
"""

from dp.modules.layers.basic_layers import conv_bn_relu
import torch.nn as nn
import torch

class DepthRendHead(nn.Module):
    def __init__(self, in_feat_channel, out_class=1, batch_norm=False, dropout_prob=0.5):
        super(DepthRendHead, self).__init__()
        self.add_cc = False
        self.add_fov = False
        self.add_nc = False
        self.add_pos_embedding = False

        self.fixed_depth_hypothesis = out_class != 1
        self.add_dep_embedding = False

        self.in_feat_channel = in_feat_channel
        self.add_chnl = 0
        if self.add_cc:
            if not self.add_pos_embedding:
                self.add_chnl += 2
        if self.add_fov:
            if not self.add_pos_embedding:
                self.add_chnl += 2
        if self.add_nc:
            if not self.add_pos_embedding:
                self.add_chnl += 2
        if not self.fixed_depth_hypothesis:
            if not self.add_dep_embedding:
                self.add_chnl += 1

        self.mlp = nn.Sequential(
            nn.Dropout2d(p=dropout_prob),
            conv_bn_relu(batch_norm, self.in_feat_channel+self.add_chnl, 512, kernel_size=1, padding=0), 
            nn.Dropout2d(p=dropout_prob),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0), 
            nn.Dropout2d(p=dropout_prob),
            conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0), 
            nn.Conv2d(512, out_class, 1), # B*out*H*W or B*out*1*N
            # nn.Sigmoid()
        )

    def forward(self, feature, depth_samples=None, cc=None, fov=None, nc=None):
        """
        feature: B*C_feature*1*N or B*C_feature*H*W
        depth_samples: B*C_samples*1*N or B*C_samples*H*W
        cc, fov, nc: see Cam-Convs
        """

        assert self.fixed_depth_hypothesis == (depth_samples is None), "{}, {}".format(self.fixed_depth_hypothesis, depth_samples)
    
        if self.add_pos_embedding:
            raise NotImplementedError
        if self.add_cc:
            assert cc is not None
            if self.add_pos_embedding:
                cc_embed = self.cc_embedding(cc)
                feature = feature + cc_embed
            else:
                feature = torch.cat([feature, cc], dim=1)
        if self.add_fov:
            assert fov is not None
            if self.add_pos_embedding:
                fov_embed = self.fov_embedding(fov)
                feature = feature + fov_embed
            else:
                feature = torch.cat([feature, fov], dim=1)
        if self.add_nc:
            assert nc is not None
            if self.add_pos_embedding:
                nc_embed = self.nc_embedding(nc)
                feature = feature + nc_embed
            else:
                feature = torch.cat([feature, nc], dim=1)
        
        if self.fixed_depth_hypothesis:
            out = self.mlp(feature)
        else:
            n_depth = depth_samples.shape[1]
            n_feat = feature.shape[1]
            feature = feature.unsqueeze(2).expand(-1, -1, n_depth, -1, -1)  # B*C_feat*C_samples*1*N

            if self.add_dep_embedding:
                dep_embed = self.dep_embedding(depth_samples)   # B*C_feat*C_samples*1*N
                feature = feature + dep_embed
                feature = feature.flatten(2, 3) # B*C_feat*NN*1
            else:
                depth_samples = depth_samples.unsqueeze(1) # B*1*C_samples*1*N
                feature = torch.cat([feature, depth_samples], dim=1)    # B*(C_feat+1)*C_samples*1*N
                feature = feature.flatten(2, 3) # B*(C_feat+1)*NN*1

            out = self.mlp(feature)
            
        return out

    def cc_embedding(self, cc):
        raise NotImplementedError

    def fov_embedding(self, fov):
        raise NotImplementedError
    
    def nc_embedding(self, nc):
        raise NotImplementedError
    
    def dep_embedding(self, dep):
        raise NotImplementedError