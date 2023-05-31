import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utilities.decode_helper import _transpose_and_gather_feat
from utilities.loss.focal_loss import focal_loss_cornernet
from utilities.loss.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from utilities.loss.dim_aware_loss import dim_aware_l1_loss
from utilities.decode_helper import decode_detections_loss


def compute_centernet3d_loss(input, target, max_objs=50, calibs=None, info=None, cls_mean_size=None, threshold=0.2):
    stats_dict = {}

    seg_loss = compute_segmentation_loss(input, target)
    offset2d_loss = compute_offset2d_loss(input, target)
    size2d_loss = compute_size2d_loss(input, target)
    offset3d_loss = compute_offset3d_loss(input, target)
    depth_loss = compute_depth_loss(input, target)
    size3d_loss = compute_size3d_loss(input, target)
    heading_loss = compute_heading_loss(input, target)
    iou3d_loss = iou_3d_loss2(input, target, calibs, cls_mean_size, info)

    # statistics
    stats_dict['seg'] = seg_loss.item()
    stats_dict['offset2d'] = offset2d_loss.item()
    stats_dict['size2d'] = size2d_loss.item()
    stats_dict['offset3d'] = offset3d_loss.item()
    stats_dict['depth'] = depth_loss.item()
    stats_dict['size3d'] = size3d_loss.item()
    stats_dict['heading'] = heading_loss.item()
    stats_dict['iou_loss'] = iou3d_loss.item()

    total_loss = seg_loss + offset2d_loss + size2d_loss + offset3d_loss + \
                 depth_loss + size3d_loss + heading_loss + iou3d_loss
    return total_loss, stats_dict


def compute_segmentation_loss(input, target):
    input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
    loss = focal_loss_cornernet(input['heatmap'], target['heatmap'])
    return loss


def compute_size2d_loss(input, target):
    # compute size2d loss
    size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
    size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
    size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
    return size2d_loss

def compute_offset2d_loss(input, target):
    # compute offset2d loss
    offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
    offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
    offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')
    return offset2d_loss


def compute_depth_loss(input, target):
    depth_input = extract_input_from_tensor(input['depth'], target['indices'], target['mask_3d'])
    depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
    depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
    depth_target = extract_target_from_tensor(target['depth'], target['mask_3d'])
    depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)
    return depth_loss


def compute_offset3d_loss(input, target):
    offset3d_input = extract_input_from_tensor(input['offset_3d'], target['indices'], target['mask_3d'])
    offset3d_target = extract_target_from_tensor(target['offset_3d'], target['mask_3d'])
    offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')
    return offset3d_loss


def compute_size3d_loss(input, target):
    size3d_input = extract_input_from_tensor(input['size_3d'], target['indices'], target['mask_3d'])
    size3d_target = extract_target_from_tensor(target['size_3d'], target['mask_3d'])
    size3d_loss = dim_aware_l1_loss(size3d_input, size3d_target, size3d_target)
    return size3d_loss


def compute_heading_loss(input, target):
    heading_input = _transpose_and_gather_feat(input['heading'], target['indices'])   # B * C * H * W ---> B * K * C
    heading_input = heading_input.view(-1, 24)
    heading_target_cls = target['heading_bin'].view(-1)
    heading_target_res = target['heading_res'].view(-1)
    mask = target['mask_2d'].view(-1)

    # classification loss
    heading_input_cls = heading_input[:, 0:12]
    heading_input_cls, heading_target_cls = heading_input_cls[mask], heading_target_cls[mask]
    if mask.sum() > 0:
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='mean')
    else:
        cls_loss = 0.0

    # regression loss
    heading_input_res = heading_input[:, 12:24]
    heading_input_res, heading_target_res = heading_input_res[mask], heading_target_res[mask]
    cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
    heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
    reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='mean')
    return cls_loss + reg_loss

def iou_3d_loss(input, target, calibs=None, cls_mean_size=None, info=None, max_obj=50, threshold=0.2):
    """
    Compute the Intersection over Union (IoU) of two 3D bounding boxes.
    
    Parameters:
        box1 (numpy.ndarray): Array representing the first 3D bounding box.
                              The shape should be (7,) where the first 6 elements
                              represent (x, y, z, w, l, h) and the last element
                              represents the yaw angle in radians.
        box2 (numpy.ndarray): Array representing the second 3D bounding box.
                              The shape should be (7,) with the same format as box1.
                              
    Returns:
        float: The IoU score between the two bounding boxes.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch, cat, height, width = input['heatmap'].size()
    topk_inds = target['indices']
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    
    
    inds = topk_inds.view(batch, max_obj)
    xs = topk_xs.view(batch, max_obj)
    ys = topk_ys.view(batch, max_obj)

    count_per_sub_bracket = torch.count_nonzero(target['mask_3d'], dim=1)

    ys = extract_target_from_tensor(ys, target['mask_3d'])
    xs = extract_target_from_tensor(xs, target['mask_3d'])
    num_objs = ys.shape[0]

    dets = []
    if num_objs > 0:

        offset_2d = extract_input_from_tensor(input['offset_2d'], inds, target['mask_3d'])
        xs2d = xs.view(num_objs, 1) + offset_2d[:, 0:1]
        ys2d = ys.view(num_objs, 1) + offset_2d[:, 1:2]
        offset_3d = extract_input_from_tensor(input['offset_3d'], inds, target['mask_3d'])
        xs3d = xs.view(num_objs, 1) + offset_3d[:, 0:1]
        ys3d = ys.view(num_objs, 1) + offset_3d[:, 1:2]
        heading = extract_input_from_tensor(input['heading'], inds, target['mask_3d'])
        depth_input = extract_input_from_tensor(input['depth'], target['indices'], target['mask_3d'])
        depth_input, _ = depth_input[:, 0:1], depth_input[:, 1:2]
        depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
        depth = depth_input

        sigma = extract_input_from_tensor(input['depth'][:, 1:2, :, :], inds, target['mask_3d'])
        size_3d = extract_input_from_tensor(input['size_3d'], inds, target['mask_3d'])
        size_2d = extract_input_from_tensor(input['size_2d'], inds, target['mask_3d'])

        cls_ids = extract_target_from_tensor(target['cls_type'], target['mask_3d'])
        scores = ys2d

        detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=1)
        dets = decode_detections_loss(dets=detections,info=info,calibs=calibs, cls_mean_size=cls_mean_size,threshold=0.2, count= count_per_sub_bracket)
    

    target_size_3D = extract_target_from_tensor(target['h_w_l'], target['mask_3d'])
    target_center_3D = extract_target_from_tensor(target['center_box_3D'], target['mask_3d'])
    target_yaw = extract_target_from_tensor(target['yaw'], target['mask_3d'])

    #iou_loss = 0
    iou_loss = torch.zeros((1,), requires_grad=True)
    iou_loss = iou_loss.to(device)

    if len(dets) != 0:
        for i, (size, center, yaw) in enumerate(zip(target_size_3D, target_center_3D, target_yaw)):
            box_pred = [dets[i][2][0],dets[i][2][1],dets[i][2][2],dets[i][1][0],dets[i][1][1],dets[i][1][2],dets[i][0]] #x,y,z,w,l,h,yaw
            box_gt = [center[0],center[1],center[2],size[0],size[1],size[2],yaw[0]] #x,y,z,w,l,h,yaw

            corners1 = get_box_corners(box_pred)
            corners2 = get_box_corners(box_gt)

            intersection_volume = compute_intersection_volume(corners1, corners2)
            box1_volume= compute_bounding_box_volume(corners1)
            box2_volume= compute_bounding_box_volume(corners2)
            
            union_volume = box1_volume + box2_volume - intersection_volume
            iou = intersection_volume / union_volume
            iou_loss += torch.ones((1,)).to(device) - iou #added terms

    if len(dets) != 0:
        iou_loss = iou_loss / len(dets)
    return iou_loss


def iou_3d_loss2(input, target, calibs=None, cls_mean_size=None, info=None, max_obj=50, threshold=0.2, diou_loss=True):
    """
    Compute the Intersection over Union (IoU) of two 3D bounding boxes.
    
    Parameters:
        box1 (numpy.ndarray): Array representing the first 3D bounding box.
                              The shape should be (7,) where the first 6 elements
                              represent (x, y, z, w, l, h) and the last element
                              represents the yaw angle in radians.
        box2 (numpy.ndarray): Array representing the second 3D bounding box.
                              The shape should be (7,) with the same format as box1.
                              
    Returns:
        float: The IoU score between the two bounding boxes.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch, cat, height, width = input['heatmap'].size()
    topk_inds = target['indices']
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    
    
    inds = topk_inds.view(batch, max_obj)
    xs = topk_xs.view(batch, max_obj)
    ys = topk_ys.view(batch, max_obj)

    count_per_sub_bracket = torch.count_nonzero(target['mask_3d'], dim=1)

    ys = extract_target_from_tensor(ys, target['mask_3d'])
    xs = extract_target_from_tensor(xs, target['mask_3d'])
    num_objs = ys.shape[0]

    dets = []
    if num_objs > 0:

        offset_2d = extract_input_from_tensor(input['offset_2d'], inds, target['mask_3d'])
        xs2d = xs.view(num_objs, 1) + offset_2d[:, 0:1]
        ys2d = ys.view(num_objs, 1) + offset_2d[:, 1:2]
        offset_3d = extract_input_from_tensor(input['offset_3d'], inds, target['mask_3d'])
        xs3d = xs.view(num_objs, 1) + offset_3d[:, 0:1]
        ys3d = ys.view(num_objs, 1) + offset_3d[:, 1:2]
        heading = extract_input_from_tensor(input['heading'], inds, target['mask_3d'])
        depth_input = extract_input_from_tensor(input['depth'], target['indices'], target['mask_3d'])
        depth_input, _ = depth_input[:, 0:1], depth_input[:, 1:2]
        depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
        depth = depth_input

        sigma = extract_input_from_tensor(input['depth'][:, 1:2, :, :], inds, target['mask_3d'])
        size_3d = extract_input_from_tensor(input['size_3d'], inds, target['mask_3d'])
        size_2d = extract_input_from_tensor(input['size_2d'], inds, target['mask_3d'])

        cls_ids = extract_target_from_tensor(target['cls_type'], target['mask_3d'])
        scores = ys2d

        detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, sigma], dim=1)
        dets = decode_detections_loss(dets=detections,info=info,calibs=calibs, cls_mean_size=cls_mean_size,threshold=0.2, count= count_per_sub_bracket)
    

    target_size_3D = extract_target_from_tensor(target['h_w_l'], target['mask_3d'])
    target_center_3D = extract_target_from_tensor(target['center_box_3D'], target['mask_3d'])
    target_yaw = extract_target_from_tensor(target['yaw'], target['mask_3d'])

    #iou_loss = 0
    iou_loss = torch.zeros((1,), requires_grad=True)
    iou_loss = iou_loss.to(device)

    if len(dets) != 0:
        for i, (size, center, yaw) in enumerate(zip(target_size_3D, target_center_3D, target_yaw)):
            box_pred = [dets[i][2][0],dets[i][2][1],dets[i][2][2],dets[i][1][0],dets[i][1][1],dets[i][1][2],dets[i][0]]
            box_gt = [center[0],center[1],center[2],size[0],size[1],size[2],yaw[0]]

            corners1 = get_box_corners(box_pred)
            corners2 = get_box_corners(box_gt)

            intersection_volume = compute_intersection_volume(corners1, corners2)
            box1_volume= compute_bounding_box_volume(corners1)
            box2_volume= compute_bounding_box_volume(corners2)
            
            union_volume = box1_volume + box2_volume - intersection_volume
            iou = intersection_volume / union_volume
            iou_loss += torch.ones((1,)).to(device) - iou
            if(diou_loss):
                iou_loss+=get_distance_centers(box_pred,box_gt)


            
            
    if len(dets) != 0:
        iou_loss = iou_loss / len(dets)
    return iou_loss

def get_distance_centers(box_pred, box_gt): 
    """
Compute the Distance-IoU (DIoU) of two 3D bounding boxes.

Parameters:
    box1 (torch.Tensor): Bounding box 1. Shape: (7).
    box2 (torch.Tensor): Bounding box 2. Shape: (7).

Returns:
    torch.Tensor: The DIoU between the two 3D bounding boxes.
"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Compute center coordinates
    center1 = torch.zeros(3, requires_grad=True)
    center2 = torch.zeros(3, requires_grad=True)

    center1 = center1.to(device)
    center2 = center2.to(device)

    center1[0] = box_pred[0]
    center1[1] = box_pred[1]
    center1[2] = box_pred[2]

    center2[0] = box_gt[0]
    center2[1] = box_gt[1]
    center2[2] = box_gt[2]

    # Compute distances
    center_distance = torch.norm(center1 - center2, p=2)
    diagonal_length = torch.norm(torch.max(center1, center2), p=2)

    # Compute DIoU
    diou_distance = (center_distance ** 2) / (diagonal_length ** 2)

    return diou_distance


######################  auxiliary functions #########################

def get_box_corners(box):
    """
    Compute the corners of a 3D bounding box.
    
    Parameters:
        box (torch.Tensor): Tensor representing the 3D bounding box.
                            The shape should be (7,) where the first 6 elements
                            represent (x, y, z, w, l, h) and the last element
                            represents the yaw angle in radians.
                             
    Returns:
        torch.Tensor: Tensor representing the corners of the bounding box.
                      The shape is (8, 3) where each row represents a corner point
                      in (x, y, z) coordinates.
    """
    # Extract box parameters
    x, y, z, w, l, h, yaw = box
    
    # Compute rotation matrix
    rotation_matrix = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Compute half-dimensions
    half_width = w / 2
    half_length = l / 2
    half_height = h / 2
    
    # Compute the 8 corners of the bounding box
    corners = torch.tensor([
        [half_width, half_length, half_height],
        [half_width, -half_length, half_height],
        [-half_width, -half_length, half_height],
        [-half_width, half_length, half_height],
        [half_width, half_length, -half_height],
        [half_width, -half_length, -half_height],
        [-half_width, -half_length, -half_height],
        [-half_width, half_length, -half_height]
    ], requires_grad=True)
    
    # Rotate and translate the corners
    rotated_corners = torch.matmul(corners, rotation_matrix.T)
    translated_corners = rotated_corners + torch.tensor([x, y, z])
    
    return translated_corners


import torch


def compute_intersection_volume(corners1, corners2):
    """
    Compute the volume of intersection between two 3D bounding boxes.

    Parameters:
        corners1 (torch.Tensor): Tensor representing the corners of the first
                                 3D bounding box. The shape should be (8, 3)
                                 where each row represents a corner point in
                                 (x, y, z) coordinates.
        corners2 (torch.Tensor): Tensor representing the corners of the second
                                 3D bounding box. The shape should be (8, 3)
                                 with the same format as corners1.

    Returns:
        torch.Tensor: The volume of intersection between the two bounding boxes.
    """
    # Compute the minimum and maximum coordinates of each box
    min_coords1 = torch.min(corners1, dim=0).values
    max_coords1 = torch.max(corners1, dim=0).values
    min_coords2 = torch.min(corners2, dim=0).values
    max_coords2 = torch.max(corners2, dim=0).values

    # Compute the intersection box dimensions
    intersection_min_coords = torch.max(min_coords1, min_coords2)
    intersection_max_coords = torch.min(max_coords1, max_coords2)
    intersection_dims = torch.clamp(intersection_max_coords - intersection_min_coords, min=0)

    # Check for non-overlapping boxes
    if torch.any(intersection_dims <= 0):
        return torch.tensor(0.0)

    # Compute the intersection volume
    intersection_volume = torch.prod(intersection_dims)
    return intersection_volume


def compute_bounding_box_volume(corners):
    """
    Compute the volume of a 3D bounding box.

    Parameters:
        corners (torch.Tensor): Tensor representing the corners of the 3D bounding box.
                                The shape should be (8, 3) where each row represents
                                a corner point in (x, y, z) coordinates.

    Returns:
        torch.Tensor: The volume of the bounding box.
    """
    # Compute the minimum and maximum coordinates of the box
    min_coords = torch.min(corners, dim=0).values
    max_coords = torch.max(corners, dim=0).values

    # Compute the dimensions of the box
    dimensions = torch.clamp(max_coords - min_coords, min=0)

    # Compute the volume of the box
    volume = dimensions[0] * dimensions[1] * dimensions[2]

    return volume


def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C

def extract_target_from_tensor(target, mask):
    return target[mask]


if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

