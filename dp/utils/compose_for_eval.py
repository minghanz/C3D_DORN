import numpy as np
import torch

def compose_preds(pred_left, pred_right, gt_width, gt_height=None):
    """This function is for composing a big depth image from two small depth images for DORN.
    pred_left, pred_right are both 2D array of H*W or 3D array of B*H*W, with no channel dimension
    inputs can be torch.Tensor or np.ndarray
    if gt_height is not given, the output height is the same as input height, otherwise the bottom part are filled
    """

    input_height = pred_left.shape[-2]
    input_width = pred_left.shape[-1]

    if gt_height is None:
        gt_height = input_height

    if pred_left.ndim == 3:
        target_shape = (pred_left.shape[0], gt_height, gt_width)
    elif pred_left.ndim == 2:
        target_shape = (gt_height, gt_width)

    overlap_width = 2*input_width-gt_width
    pred_overlap = 0.5 * (pred_left[..., input_width-overlap_width:] + pred_right[..., :overlap_width])

    if isinstance(pred_left, np.ndarray):
        pred_full = np.zeros(target_shape, dtype=pred_left.dtype)
    elif isinstance(pred_left, torch.Tensor):
        pred_full = torch.zeros(target_shape, dtype=pred_left.dtype)
    else:
        raise ValueError("type of input not recognized", type(pred_left), type(pred_right) )

    pred_full[..., gt_height-input_height:gt_height, :input_width-overlap_width] = pred_left[..., :input_width-overlap_width]
    pred_full[..., gt_height-input_height:gt_height, input_width:] = pred_right[..., overlap_width:]
    pred_full[..., gt_height-input_height:gt_height, input_width-overlap_width:input_width] = pred_overlap

    return pred_full

def kb_crop_preds(pred, mode="kb_crop"):
    """
    input B*H*W
    output pred of (B*)352*1216, gt: full image size
    """
    if mode == "kb_crop":
        kb_width = 1216
        kb_height = 352
    elif mode == "vkitti2":
        kb_width = pred.shape[-1]
        kb_height = int(0.5 * pred.shape[-2])
    else:
        raise Value("mode {} not recognized".format(mode))

    left_margin = int(0.5*(pred.shape[-1] - kb_width))
    top_margin = pred.shape[-2] - kb_height
    pred = pred[..., top_margin:, left_margin:left_margin+kb_width]

    return pred