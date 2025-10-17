"""ByteFormer 版本 COCO 数据加载封装。
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Sequence, Tuple

import torch
import numpy as np
from torchvision import transforms  # noqa: F401 (预留未来扩展)

from PureT.lib.config import cfg
from PureT.datasets_.coco_dataset_hf import CocoDataset
import PureT.samplers.distributed as distributed_samplers
from corenet.data.collate_fns.byteformer_collate_functions import byteformer_image_collate_fn
from PureT.byteformer_immigration import get_opts

opts = get_opts()

# 确保 opts 中包含 image_augmentation.pil_save.corrupt_level 的默认值，
# 以便上层注入或默认 "none" 时 transform 能安全读取。
import argparse as _argparse
if not hasattr(opts, "image_augmentation"):
    opts.image_augmentation = _argparse.Namespace()
if not hasattr(opts.image_augmentation, "pil_save"):
    opts.image_augmentation.pil_save = _argparse.Namespace()
if not hasattr(opts.image_augmentation.pil_save, "corrupt_level"):
    opts.image_augmentation.pil_save.corrupt_level = "none"

def sample_collate(batch):
    indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    """
    # 读取图像的预训练特征时，大小为[L, D]，其中L的长度可能不一（如目标特征）
    # 因此需要进行特征数量判断，并生成特征掩码 att_mask
    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)
    """
    # 图像特征，无需与预训练特征一样进行特征数量判断，直接合并即可
    # att_mask为最终grid特征大小，实际上grid特征无需att_mask亦可
    att_feats = torch.stack(att_feats, 0)  # [B, 3, 384, 384]
    att_mask = torch.ones(att_feats.size()[0], 12*12)

    return indices, input_seq, target_seq, gv_feat, att_feats, att_mask

def sample_collate_val(batch):
    indices, gv_feat, att_feats = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    """
    # 读取图像的预训练特征时，大小为[L, D]，其中L的长度可能不一（如目标特征）
    # 因此需要进行特征数量判断，并生成特征掩码 att_mask
    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)
    """
    # 图像特征，无需与预训练特征一样进行特征数量判断，直接合并即可
    # att_mask为最终grid特征大小，实际上grid特征无需att_mask亦可
    att_feats = torch.stack(att_feats, 0)  # [B, 3, 384, 384]
    att_mask = torch.ones(att_feats.size()[0], 12*12)

    return indices, gv_feat, att_feats, att_mask

def byteformer_collate(batch: Sequence[Tuple[Any, ...]]):
    """
    训练阶段 collate。
    """
    indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    """
    # 读取图像的预训练特征时，大小为[L, D]，其中L的长度可能不一（如目标特征）
    # 因此需要进行特征数量判断，并生成特征掩码 att_mask
    atts_num = [x.shape[0] for x in att_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    att_feats = torch.cat(feat_arr, 0)
    att_mask = torch.cat(mask_arr, 0)
    """
    # 图像特征，无需与预训练特征一样进行特征数量判断，直接合并即可
    # att_mask为最终grid特征大小，实际上grid特征无需att_mask亦可  
    
    att_feats = torch.stack(att_feats, 0)  # [B, 3, 224, 224]
    
    corenet_batch = []
    for img_tensor in att_feats:
        corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target
    collated = byteformer_image_collate_fn(corenet_batch, opts)
    att_feats = collated["samples"]
    att_mask = None
    
    return indices, input_seq, target_seq, gv_feat, att_feats, att_mask

def byteformer_collate_val(batch: Sequence[Tuple[Any, ...]]):
    """验证阶段 collate。
    已修改以支持样本堆叠增强，会同步复制所有元数据。
    """
    indices, gv_feat, att_feats = zip(*batch)
    
    # 1. 将图像张量打包成 corenet 期望的格式
    corenet_batch = []
    for img_tensor in att_feats:
        corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})  # dummy target

    # 2. 调用核心 collate 函数，这将执行 1->N 的增强
    collated = byteformer_image_collate_fn(corenet_batch, opts)
    augmented_att_feats = collated["samples"]
    
    # 3. 计算增强因子（例如，4）
    original_bs = len(att_feats)
    if original_bs == 0:
        # 处理空批次的情况
        return torch.tensor(indices), torch.tensor(gv_feat), augmented_att_feats, None

    augmentation_factor = augmented_att_feats.size(0) // original_bs
    
    # 如果没有增强，直接返回
    if augmentation_factor <= 1:
        indices = np.stack(indices, axis=0).reshape(-1)
        gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
        att_mask = None # ByteFormer collate 后通常为 None
        return indices, gv_feat, augmented_att_feats, att_mask

    # 4. 关键修复：将所有元数据复制 N 份以匹配增强后的数据
    print(f"[DEBUG Metadata] Augmentation factor is {augmentation_factor}. Duplicating metadata.")
    
    # 复制 indices
    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)
    
    # 复制 gv_feat
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    # gv_feat 的形状是 [B, D]，我们需要将其扩展为 [B*N, D]
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    att_mask = None # ByteFormer collate 后通常为 None

    return expanded_indices, expanded_gv_feat, augmented_att_feats, att_mask


def _worker_init_fn(worker_id: int) -> None:
    """为每个 DataLoader worker 设置独立但可复现的随机种子。"""
    base_seed = torch.initial_seed() % 2**31
    np.random.seed(base_seed + worker_id)
    import random as _random
    _random.seed(base_seed + worker_id)


def load_train(distributed: bool, epoch: int, coco_set: CocoDataset):
    """构建训练 DataLoader。

    参数：
        distributed: 是否分布式
        epoch: 当前 epoch (用于分布式 sampler 设置 shuffle seed)
        coco_set: 已实例化的 CocoDataset
    """
    sampler = distributed_samplers.DistributedSampler(coco_set, epoch=epoch) if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if cfg.DATA_LOADER.NUM_WORKERS > 0 else False,
        collate_fn=byteformer_collate,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    return loader

def load_val(image_ids_path, gv_feat_path: str = '', att_feats_folder=None, max_samples: int = 200):  # noqa: D401
    """构建验证 DataLoader（进入数据集 validation 模式）。"""
    coco_set = CocoDataset(
        image_ids_path=image_ids_path,
        input_seq=None,  # None 触发 validation mode
        target_seq=None,
        gv_feat_path=gv_feat_path or '',
        seq_per_img=1,
        max_feat_num=cfg.DATA_LOADER.MAX_FEAT,
        max_samples=max_samples,
    )
    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if cfg.DATA_LOADER.NUM_WORKERS > 0 else False,
        collate_fn=byteformer_collate_val,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    return loader
