import torch, os
import numpy as np
import torchvision.transforms as transforms
import predictors.Lane_width_Testing.data.mytransforms as mytransforms
from predictors.Lane_width_Testing.data.constant import tusimple_row_anchor, culane_row_anchor
from predictors.Lane_width_Testing.data.dataset import LaneClsDataset, LaneTestDataset

def get_train_loader(batch_size, data_root, griding_num, dataset, use_aux, num_lanes):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((36, 100)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    
    if dataset == 'CULane':
        train_dataset = LaneClsDataset(
            data_root,
            os.path.join(data_root, 'list/train_gt.txt').replace("\\", "/"),
            img_transform=img_transform, target_transform=target_transform,
            simu_transform=simu_transform,
            segment_transform=segment_transform, 
            row_anchor=culane_row_anchor,
            griding_num=griding_num, use_aux=use_aux, num_lanes=num_lanes
        )
        cls_num_per_lane = 18

    elif dataset == 'Tusimple':
        train_dataset = LaneClsDataset(
            data_root,
            os.path.join(data_root, 'train_gt.txt').replace("\\", "/"),
            img_transform=img_transform, target_transform=target_transform,
            simu_transform=simu_transform,
            griding_num=griding_num, 
            row_anchor=tusimple_row_anchor,
            segment_transform=segment_transform, use_aux=use_aux, num_lanes=num_lanes
        )
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    # Disable multi-threading or parallelism by setting num_workers=0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=None, num_workers=0, pin_memory=False
    )

    return train_loader, cls_num_per_lane

def get_test_loader(batch_size, data_root, dataset):
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_dataset = LaneTestDataset(
        data_root,
        os.path.join(data_root, 'test.txt').replace("\\", "/"),
        img_transform=img_transforms
    )
    cls_num_per_lane = 56

    # Disable multi-threading or parallelism by setting num_workers=0
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=None, num_workers=0, pin_memory=False
    )
    return loader
