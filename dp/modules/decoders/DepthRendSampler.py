"""
This module is to sample depth values from the probability vector of predefined intervals 
(For each pixel, there is a vector of length ord_num representing the proability of the depth falling into a bin). 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthRendSampler(nn.Module):
    def __init__(self, n_dep_sample=30, n_pt_sample=900, align_corners=False):
        super(DepthRendSampler, self).__init__()

        self.n_dep_sample = n_dep_sample
        self.flag_dep_sample = n_dep_sample > 0
        if self.flag_dep_sample:
            raise NotImplementedError

        self.n_pt_sample = n_pt_sample
        self.flag_pt_sample = n_pt_sample > 0
        assert self.n_pt_sample % 3 ==0, self.n_pt_sample   # 3 parts: max entropy, max laplacian of entropy, random
        self.n_pt_samp_each = self.n_pt_sample // 3
        self.n_pt_samp_each_pool = self.n_pt_samp_each * 2

        self.conv2d_laplace = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1,1,3,bias=False)
        )
        laplace_kernel = torch.tensor([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]], dtype=torch.float32).expand(1, 1, 3, 3)
        self.conv2d_laplace[1].weight = torch.nn.Parameter(-laplace_kernel, requires_grad=False)    # negative sign so that the filtered map max at local maxima of entropy
        # ### or 
        # self.conv2d_laplace[1].weight.data = laplace_kernel
        # self.conv2d_laplace[1].weight.requires_grad = False

        self.align_corners = align_corners

    def apply_sampling(self, x_dict, idxs, H=None, W=None):
        
        if idxs.ndim == 4:  # B*1*n_samp*2
            assert H is not None
            assert W is not None
            ### The indices are float. Use F.grid_sample
            idx_hw = idxs
            ### select the elements based on the indexes
            idx_hw_normalize = self.idx_hw_to_normalize(idx_hw, H, W)

            sampled_dict = dict()
            for key, item in x_dict.items():
                sampled_dict[key] = F.grid_sample(item, idx_hw_normalize, align_corners=self.align_corners)  # B*C*1*n_samp_all
        else:               # B*1*n_samp
            ### The indices are int. Use torch.gather (No build-in function to sample on two-dim simultaneously. )
            idx_flat = idxs
            sampled_dict = dict()
            for key, item in x_dict.items():
                item_flat_sampled = self.sampling_get_from_idx_flat(item, idx_flat)
                sampled_dict[key] = item_flat_sampled

        return sampled_dict

    def sampling_set_from_idx_flat(self, x, idx_flat, src_flat):
        """
        x: B*C*H*W, idx_flat: B*1*n_samp, src_flat: B*C*1*n_samp
        
        """
        ### put the x into the right place of upsampled p_bin
        shape_x = x.shape
        src_flat = src_flat.squeeze(2)                                              # B*C*n_samp
        x_flat = x.flatten(2, 3)                                                    # B*C*N
        idx_flat_expand = idx_flat.expand(-1, x_flat.shape[1], -1)                  # B*C*n_samp
        x_flat = torch.scatter(x_flat, dim=2, index=idx_flat_expand, src=src_flat)            # B*C*N
        x = x_flat.reshape(shape_x)
        return x

    def sampling_get_from_idx_flat(self, x, idx_flat):
        """
        x: B*C*H*W, idx_flat: B*1*n_samp
        return: B*C*1*n_samp
        """
        x_flat = x.flatten(2, 3)                                                    # B*C*N
        idx_flat_expand = idx_flat.expand(-1, x_flat.shape[1], -1)                  # B*C*n_samp
        x_flat_sampled = torch.gather(x_flat, dim=2, index=idx_flat_expand)         # B*C*n_samp
        x_flat_sampled = x_flat_sampled.unsqueeze(2)                                # B*C*1*n_samp
        return x_flat_sampled

    def gen_sampling(self, p_bin, mask=None, nsamp_downscale=0):
        """
        p_bin are after softmax. B*ord_num*H*W
        mask (optional): B*1*H_full*W_full binary mask, 1's are the pixels which can be selected. This can be of different size as p_bin and the wanted
        return: indices of sampled points, and values of sampled depth
        """
        B, ord_num, H, W = p_bin.shape

        n_samp = self.n_pt_samp_each // (2**nsamp_downscale)

        ### evaluate Shannon entropy
        p_bin_clamped = torch.clamp(p_bin, min=1e-6, max=1-1e-6)
        entropy = - (p_bin_clamped * torch.log(p_bin_clamped)).sum(1, keepdim=True) # B*1*H*W
        
        ### Laplacian of entropy (local minima and maxima)
        entropy_laplace = self.conv2d_laplace(entropy) # B*1*H*W

        if self.flag_pt_sample:
            if self.training and mask is not None:
                _, _, H_full, W_full = mask.shape
                entropy_full = F.interpolate(entropy, (H_full, W_full), mode="bilinear", align_corners=self.align_corners)
                entropy_laplace_full = F.interpolate(entropy_laplace, (H_full, W_full), mode="bilinear", align_corners=self.align_corners)

                dummy_neg = - torch.ones_like(entropy_full) * float("Inf")
                entropy_full = torch.where(mask, entropy_full, dummy_neg)
                entropy_laplace_full = torch.where(mask, entropy_laplace_full, dummy_neg)

                entropy_flat_full = entropy_full.flatten(2,3)
                entropy_laplace_flat_full = entropy_laplace_full.flatten(2,3)

                ent_idx_flat_full = self.samp_idx_from_value_top(entropy_flat_full, n_samp)         # B*1*n_samp
                lap_idx_flat_full = self.samp_idx_from_value_top(entropy_laplace_flat_full, n_samp)
                
                ent_idx_hw_full = self.idx_flat_to_hw(ent_idx_flat_full, H_full, W_full)    # B*1*n_samp*2
                lap_idx_hw_full = self.idx_flat_to_hw(lap_idx_flat_full, H_full, W_full)
                
                rnd_idx_hw_full = self.samp_idx_from_mask_rnd(mask, n_samp)                         # B*1*n_samp*2
                rnd_idx_flat_full = self.idx_hw_to_flat(rnd_idx_hw_full, H_full, W_full)    # B*1*n_samp

                idx_hw_full = torch.cat([ent_idx_hw_full, lap_idx_hw_full, rnd_idx_hw_full], dim=2) # B*1*n_samp_all*2

                idx_hw_normalize = self.idx_hw_to_normalize(idx_hw_full, H_full, W_full)
                idx_hw = self.idx_normalize_to_hw(idx_hw_normalize, H, W)

            else:
                ### pick topk and random sampling from topk (TODO: make sure the points are not duplicated across sets)
                entropy_flat = entropy.flatten(2,3) # B*1*N
                entropy_laplace_flat = entropy_laplace.flatten(2,3)

                ent_idx_flat = self.samp_idx_from_value_top(entropy_flat, n_samp)           # B*1*n_samp
                lap_idx_flat = self.samp_idx_from_value_top(entropy_laplace_flat, n_samp)
                rnd_idx_flat = self.samp_idx_from_value_rnd(entropy_flat, n_samp)

                idx_flat = torch.cat([ent_idx_flat, lap_idx_flat, rnd_idx_flat], dim=2)    # B*1*n_samp_all

                idx_hw = self.idx_flat_to_hw(idx_flat, H, W)    # B*1*n_samp*2
                ### if in training, should return float coordinates. In inference, sample on grids. 
                if self.training:
                    idx_hw = idx_hw + torch.rand(idx_hw.shape, device=idx_hw.device)-0.5

            # return idx_hw
            if self.training:
                return idx_hw
            else:
                return idx_hw, idx_flat

    def idx_normalize_to_hw(self, idx_hw_normalize, h, w):
        idx_hw_normalize = (idx_hw_normalize + 1) * 0.5 # [0, 1]
        if self.align_corners:
            idx_h = idx_hw_normalize[..., 0] * (h-1)
            idx_w = idx_hw_normalize[..., 1] * (w-1)
        else:
            idx_h = idx_hw_normalize[..., 0] * h - 0.5
            idx_w = idx_hw_normalize[..., 0] * w - 0.5

        idx_hw = torch.stack([idx_h, idx_w], dim=3) # # B*1*n_samp*2
        return idx_hw

    def idx_hw_to_normalize(self, idx_hw, h, w):
        idx_hw = idx_hw.float()
        if self.align_corners:
            idx_h_normalize = idx_hw[..., 0] / (h-1) * 2 - 1    # [-1, 1]
            idx_w_normalize = idx_hw[..., 1] / (w-1) * 2 - 1
        else:
            idx_h_normalize = (idx_hw[..., 0] + 0.5) / h * 2 - 1 # [-1, 1]
            idx_w_normalize = (idx_hw[..., 1] + 0.5) / w * 2 - 1 # [-1, 1]

        idx_hw_normalize = torch.stack([idx_h_normalize, idx_w_normalize], dim=3)    # B*1*n_samp*2
        return idx_hw_normalize

    def idx_hw_to_flat(self, idx_hw, h, w):
        idx_flat = idx_hw[..., 0] * w + idx_hw[..., 1]  # B*1*n_samp
        return idx_flat

    def idx_flat_to_hw(self, idx_flat, h, w):
        idx_h = idx_flat // w
        idx_w = idx_flat % w
        idx_hw = torch.stack([idx_h, idx_w], dim=3) # B*1*n_samp*2
        return idx_hw

    def samp_idx_from_mask_rnd(self, mask, n_samp):
        B, C, H, W = mask.shape
        idx_list = []
        for i in range(B):
            idx_i = mask[i, 0].nonzero()       # n_nz * 2
            n_idx_i = idx_i.shape[0]
            idx_of_idx_i = torch.randperm(n_idx_i, device=mask.device)[:n_samp]    # n_samp
            idx_list.append(idx_i[idx_of_idx_i])    # n_samp * 2

        idx = torch.stack(idx_list, dim=0).unsqueeze(1)       # B*1*n_samp*2
        return idx

    def samp_idx_from_value_rnd(self, value_flat, n_samp):
        n_pt_total = value_flat.shape[2]
        idx_list = []
        for i in range(value_flat.shape[0]):
            idx_list.append(torch.randperm(n_pt_total, device=value_flat.device)[:n_samp])
        ent_idx = torch.stack(idx_list, dim=0).unsqueeze(1)    # B*1*n_samp
        
        return ent_idx

    def samp_idx_from_value_top(self, value_flat, n_samp):
        n_samp_pool = n_samp * 2
        ent_topk = torch.topk(value_flat, n_samp_pool, dim=2) # values: B*1*n_pool, indices: B*1*n_pool
        ent_idx_of_idx = torch.randperm(n_samp_pool, device=value_flat.device)[:n_samp] # n_samp
        ent_idx_of_idx = ent_idx_of_idx.expand(value_flat.shape[0], 1, -1)           # B*1*n_samp
        ent_idx = torch.gather(ent_topk.indices, dim=2, index=ent_idx_of_idx)   # B*1*n_samp

        return ent_idx