"""
The class is responsible for sampling, upsampling, per-point rendering, and merge the result into the original depth prediction. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dp.modules.decoders.DepthRendSampler import DepthRendSampler
from dp.modules.decoders.DepthRendHead import DepthRendHead

class DepthRend(nn.Module):
    def __init__(self, n_dep_sample=30, n_pt_sample=900, in_feat_channel=512*5, out_class=90, size=(385, 513), align_corners=False, batch_norm=False, dropout_prob=0.5):
        super(DepthRend, self).__init__()
        self.align_corners = align_corners

        self.sampler = DepthRendSampler(n_dep_sample, n_pt_sample, align_corners)
        self.mlp = DepthRendHead(in_feat_channel, out_class, batch_norm, dropout_prob=dropout_prob)

        self.size = size

    def forward(self, x_dict, target=None, mask=None, nsamp_downscale=3):
        """
        p_bin and feature: low resolution. 
        mask: high (original) resolution, because it is sparse and resizing could cause problem. 
        return: logits (unnormalized, before soft max), since the output of SceneUnderstandingModule is also unnormalized. 
        """

        if self.training:
            assert mask is not None
            pmul = x_dict["full"]["pmul"]
            feature = x_dict["net"]["feat"]
            ### directly interpolate the low-res depth to high-res
            ### The sampled locations are supervised separately (or replacing the corresponding high-res location?)
            idx_hw = self.sampler.gen_sampling(pmul, mask, nsamp_downscale=1)
            H = pmul.shape[2]
            W = pmul.shape[3]
            feature_dict = {"feature": feature, "target": target.unsqueeze(1)}
            feat_samp_dict = self.sampler.apply_sampling(feature_dict, idx_hw, H, W)
            x = self.mlp(feat_samp_dict['feature'])
            
            x_dict["samp"] = dict()
            x_dict["samp"]["logit"] = x
            x_dict["samp"]["target"] = feat_samp_dict['target'].squeeze(1)
        else:
            logit = x_dict["net"]["logit"]
            feature = x_dict["net"]["feat"]
            sample_painter = torch.zeros((logit.shape[0], 1, logit.shape[2], logit.shape[3]), dtype=logit.dtype, device=logit.device)

            ### recursively upsample the low-res depth prediction to high-res depth prediction
            for scale in range(nsamp_downscale-1, -1, -1):    # nsamp_downscale-1, ..., 0
                sample_painter = F.interpolate(sample_painter, (logit.shape[2]*2, logit.shape[3]*2), mode='bilinear', align_corners=self.align_corners )  # B*ord_num*H*W
                logit = F.interpolate(logit, (logit.shape[2]*2, logit.shape[3]*2), mode='bilinear', align_corners=self.align_corners )  # B*ord_num*H*W
                pmul = F.softmax(logit, dim=1)

                # p_bin = F.interpolate(p_bin, (p_bin.shape[2]*2, p_bin.shape[3]*2), mode='bilinear', align_corners=self.align_corners )  # B*ord_num*H*W
                idx_hw, idx_flat = self.sampler.gen_sampling(pmul, mask=None, nsamp_downscale=scale)     # B*1*n_samp*2, B*1*n_samp
                H = pmul.shape[2]
                W = pmul.shape[3]
                feature_dict = {"feature": feature}
                feat_samp_dict = self.sampler.apply_sampling(feature_dict, idx_hw, H, W)
                x = self.mlp(feat_samp_dict['feature'])                     # B*ord_num*1*n_samp
                ### put the x into the right place of upsampled p_bin
                logit = self.sampler.sampling_set_from_idx_flat(logit, idx_flat, x)

                sample_dots = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
                sample_painter = self.sampler.sampling_set_from_idx_flat(sample_painter, idx_flat, sample_dots)

            assert logit.shape[2] == self.size[0]
            assert logit.shape[3] == self.size[1]

            x_dict["full"]["logit"] = logit
            x_dict["full"]["sample_painter"] = sample_painter

        return #x_dict