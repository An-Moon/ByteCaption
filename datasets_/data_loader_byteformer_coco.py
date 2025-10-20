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

# 加了一个BLIP专用的collate函数，导入这些模块，先放到这里
import io
from PIL import Image
from corenet.data.transforms import image_bytes
from torchvision import transforms as T # 使用别名避免冲突

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
    """验证阶段 collate。"""
    indices, gv_feat, att_feats = zip(*batch)
    
    original_indices = np.array(indices)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    att_feats = torch.stack(att_feats, 0)

    corenet_batch = []
    for img_tensor in att_feats:
        corenet_batch.append({"samples": img_tensor, "targets": torch.tensor(0)})
    
    # 调用 collate 函数，它会进行数据增强
    # 注意：这里不再需要 deepcopy(opts)，因为我们不会在循环外修改它
    collated = byteformer_image_collate_fn(corenet_batch, opts)
    
    # 获取增强后的图像特征
    final_att_feats = collated["samples"]
    att_mask = None # ByteFormer 通常在模型内部处理掩码

    # 计算增强因子 (e.g., 4)
    expansion_factor = final_att_feats.size(0) // len(original_indices)
    
    if expansion_factor > 1:
        # 1. 创建唯一的ID (保持现有逻辑)
        unique_indices = []
        for i in original_indices:
            for aug_idx in range(expansion_factor):
                unique_indices.append(i * 100 + aug_idx)
        final_indices = np.array(unique_indices)
        
        # 2. 同步扩展全局特征 gv_feat
        final_gv_feat = gv_feat.repeat_interleave(expansion_factor, dim=0)
    else:
        # 如果没有增强，则一切保持原样
        final_indices = original_indices
        final_gv_feat = gv_feat

    # 返回一个完全对齐的元组
    return final_indices, final_gv_feat, final_att_feats, att_mask

def blip_collate_val(batch: Sequence[Tuple[Any, ...]]):
    """
    验证阶段 collate，专门为 BLIP 模型准备数据。
    它会模拟 ByteFormer 的损坏流程，并尝试解码图像。
    """
    indices, gv_feat, att_feats = zip(*batch)
    
    # 1. 初始化 BLIP 的图像处理器和 ByteFormer 的损坏器
    blip_image_tensors = []
    corrupter = image_bytes.ByteStreamCorrupter(opts)
    # 这是标准的 BLIP 预处理流程
    blip_transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # 2. 对批次中的每个原始图像执行“编码 -> 损坏 -> 解码”流程
    for img_tensor in att_feats:
        try:
            byte_stream = image_bytes._image_to_bytes(img_tensor, format="jpeg", quality=95)
            original_bytes = byte_stream.getvalue()
        except Exception:
            num_corruptions = len(corrupter.corruption_types) if corrupter.corruption_types else 1
            blip_image_tensors.extend([None] * num_corruptions)
            continue

        corruption_types_to_apply = corrupter.corruption_types if corrupter.corruption_types else ["none"]
        for corruption_type in corruption_types_to_apply:
            corrupted_bytes = original_bytes
            if corruption_type != "none":
                if corruption_type == "bit_flip":
                    corrupted_bytes = corrupter._random_bit_flip(original_bytes, corrupter.params["bit_flip"])
                elif corruption_type == "segment_dropout":
                    corrupted_bytes = corrupter._segment_dropout(original_bytes, corrupter.params["drop"])
                elif corruption_type == "header_truncation":
                    corrupted_bytes = corrupter._header_truncation(original_bytes, corrupter.params["head"])
                elif corruption_type == "tail_truncation":
                    corrupted_bytes = corrupter._tail_truncation(original_bytes, corrupter.params["tail"])
            
            try:
                reconstructed_img = Image.open(io.BytesIO(corrupted_bytes)).convert("RGB")
                # blip_ready_tensor = blip_transform(reconstructed_img)
                blip_image_tensors.append(reconstructed_img)
            except Exception:
                # 解码失败，用 None 作为占位符
                blip_image_tensors.append(None)

    # 3. 同步元数据，使其数量与增强后的图像数量匹配
    original_bs = len(att_feats)
    augmentation_factor = len(blip_image_tensors) // original_bs if original_bs > 0 else 1
    
    indices_np = np.stack(indices, axis=0).reshape(-1)
    expanded_indices = np.repeat(indices_np, augmentation_factor, axis=0)
    
    gv_feat_tensor = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)
    expanded_gv_feat = gv_feat_tensor.repeat_interleave(augmentation_factor, dim=0)

    # 我们将 BLIP 的数据放在原本 att_feats 的位置，以保持返回结构一致
    return expanded_indices, expanded_gv_feat, blip_image_tensors, None

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

def load_val(image_ids_path, gv_feat_path: str = '', att_feats_folder=None, max_samples: int = 200, eval_mode='byteformer'):  # noqa: D401
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
    # 加一个选择，先这样加，之后再说
    if eval_mode == 'byteformer':
        active_collate_fn = byteformer_collate_val
    elif eval_mode == 'blip':
        active_collate_fn = blip_collate_val
    else:
        print(f"[错误] 未知的 eval_mode: '{eval_mode}'。将回退到默认的 'byteformer' 模式。")
        active_collate_fn = byteformer_collate_val # <-- 明确设置一个默认值


    loader = torch.utils.data.DataLoader(
        coco_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if cfg.DATA_LOADER.NUM_WORKERS > 0 else False,
        collate_fn=active_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
    )
    return loader