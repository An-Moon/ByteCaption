import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler_coco import CocoEvaler
from evaluation.evaler_flickr8k import Flickr8kEvaler
from scorer.coco_scorer import CocoScorer
from scorer.flickr8k_scorer import Flickr8kScorer
from scorer.scorer import Scorer  # 新增导入
from lib.config import cfg, cfg_from_file

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from main import init_distributed_mode

"""
cd /root/autodl-tmp/ByteCaption && PYTHONPATH=/root/autodl-tmp/ByteCaption python PureT/main_val.py --folder PureT/experiments/ByteCaption_XE --val_samples 500 --resume -1   --disable_wandb
"""

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args

        # byteformer/blip
        self.eval_mode = 'blip'
        self.corrupt_level = getattr(args, "corrupt_level", "none")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据集类型（与训练脚本保持一致）
        self.dataset_type = getattr(args, "dataset", "coco").lower()

        # 固定随机数种子以便可复现
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            np.random.seed(int(cfg.SEED))
            torch.manual_seed(int(cfg.SEED))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(cfg.SEED))

        self.distributed = torch.cuda.device_count() > 1 and torch.distributed.is_available()
        
        if self.distributed:
            self.local_rank = init_distributed_mode()
        else:
            self.local_rank = 0
        self.is_master = (not self.distributed) or (dist.get_rank() == 0 if self.distributed else True)

        # 提供简单的 config summary / logging helper，避免未定义调用导致崩溃
        self._print_eval_summary(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.setup_logging()
        self.setup_wandb()
        self.setup_network()

        # 根据数据集类型创建评估器（与 main.py 对齐）
        val_samples = getattr(args, "val_samples", 100)
        if val_samples == 0:
            val_samples = None

        enable_eval_loss = True

        # 先不依赖 training_dataset（评估脚本通常不需要训练集引用）
        if self.dataset_type == "coco":
            # 不使用未定义的 self.training_dataset，传 None
            self.scorer = CocoScorer(shared_dataset=None)
            eval_ids_path = cfg.DATA_LOADER.VAL_ID if cfg.DATA_LOADER.VAL_ID else None
            val_annfile = cfg.INFERENCE.VAL_ANNFILE
            self.val_evaler = CocoEvaler(
                eval_ids_path,
                cfg.DATA_LOADER.VAL_GV_FEAT,
                cfg.DATA_LOADER.VAL_ATT_FEATS,
                val_annfile,
                max_samples=val_samples,
                enable_eval_loss=enable_eval_loss,
                eval_mode=self.eval_mode,
            )
            self._log(f"Validation dataset (COCO): Using {val_samples if val_samples else 'ALL'} samples", prefix="DATASET")
        elif self.dataset_type == "flickr8k":
            self.scorer = Flickr8kScorer(shared_dataset=None)
            eval_ids_path = cfg.DATA_LOADER.VAL_ID if cfg.DATA_LOADER.VAL_ID else None
            val_annfile = cfg.INFERENCE.VAL_ANNFILE
            self.val_evaler = Flickr8kEvaler(
                eval_ids_path,
                cfg.DATA_LOADER.VAL_GV_FEAT,
                cfg.DATA_LOADER.VAL_ATT_FEATS,
                val_annfile,
                max_samples=val_samples,
                enable_eval_loss=enable_eval_loss,
                corrupt_level=self.corrupt_level,
            )
            self._log(f"Validation dataset (Flickr8k): Using {val_samples if val_samples else 'ALL'} samples", prefix="DATASET")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        # 可选地创建 test_evaler（保持和 main.py 一致，此处默认不启用以节省时间）
        self.test_evaler = None

        # 选择 scorer（仅用于训练时的 reward 计算；评估脚本保留）
        if self.dataset_type == "coco":
            self.scorer = CocoScorer(shared_dataset=None)
        elif self.dataset_type == "flickr8k":
            self.scorer = Flickr8kScorer(shared_dataset=None)
        else:
            self.scorer = Scorer()

    def setup_wandb(self):
        """Initializes wandb if enabled."""
        self.use_wandb = WANDB_AVAILABLE and not self.args.disable_wandb
        if self.is_master and self.use_wandb:
            # Generate a run name if not provided
            if self.args.wandb_name:
                run_name = self.args.wandb_name
            else:
                # e.g., eval-best-corrupt_light
                model_id = 'best' if self.args.resume == -1 else f'epoch_{self.args.resume}'
                run_name = f"eval-{model_id}-corrupt_{self.corrupt_level}"

            wandb.init(
                project=self.args.wandb_project,
                name=run_name,
                config=vars(self.args)  # Log all command-line arguments
            )
            wandb.config.update(cfg, allow_val_change=True) # Log yml config
            self._log("Wandb logging is ENABLED.", prefix="WANDB")
        elif self.is_master:
            self._log("Wandb logging is DISABLED.", prefix="WANDB")

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        # 避免重复添加handler
        if not self.logger.handlers:
            self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, 'OfflineVal_' + cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(model).cuda()
        else:
            self.model = model.to(self.device)

        if self.args.resume > 0:
            ckpt = self.snapshot_path("caption_model", self.args.resume)
            if os.path.exists(ckpt):
                self.model.load_state_dict(
                    torch.load(ckpt, map_location=lambda storage, loc: storage)
                )
                self.logger.info(f"Loaded checkpoint: {ckpt}")
            else:
                self.logger.warning(f"Requested checkpoint for resume not found: {ckpt}")
        elif self.args.resume == -1:
            # 使用 cfg.ROOT_DIR 下 snapshot/best_model.pth，避免硬编码路径
            best_ckpt = os.path.join(cfg.ROOT_DIR or self.args.folder or ".", "snapshot", "best_model.pth")
            if os.path.exists(best_ckpt):
                self.model.load_state_dict(
                    torch.load(best_ckpt, map_location=lambda storage, loc: storage)
                )
                self.logger.info(f"Loaded best model: {best_ckpt}")
            else:
                self.logger.warning(f"best_model.pth not found at: {best_ckpt}")


    def eval(self, epoch):
        # 记录并打印评估结果
        if self.val_evaler is not None:
            val_res = self.val_evaler(self.model, 'val_' + str(epoch))
            self.logger.info('######## Offline VAL ' + str(epoch) + ' ########')
            self.logger.info(str(val_res))
            if self.is_master and self.use_wandb:
                wandb.log({f"eval/{k}": v for k, v in val_res.items()})
        else:
            self.logger.info('VAL evaluation skipped (no val_evaler).')

        if self.test_evaler is not None:
            test_res = self.test_evaler(self.model, 'test_' + str(epoch))
            self.logger.info('######## Offline TEST ' + str(epoch) + ' ########')
            self.logger.info(str(test_res))
        else:
            self.logger.info('TEST evaluation skipped (no test_evaler).')

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def _print_eval_summary(self, args):
        """打印评估相关的摘要（在 Tester 初始化后调用）"""
        if not self.is_master:
            return

        self._log_section("EVALUATION CONFIGURATION")
        print(f"  Dataset Type         : {self.dataset_type.upper()}")
        print(f"  Corrupt Level        : {getattr(self, 'corrupt_level', 'none')}")
        resume = getattr(args, "resume", -1)
        mode = "best (--resume==-1)" if resume == -1 else ("checkpoint" if resume > 0 else "auto-latest")
        print(f"  Evaluation Mode      : {mode} (resume={resume})")
        best_path = os.path.join(cfg.ROOT_DIR or args.folder or ".", "snapshot", "best_model.pth")
        print(f"  Best model path      : {best_path}")
        val_samples = getattr(args, "val_samples", 0)
        print(f"  Validation Samples   : {val_samples if val_samples > 0 else 'ALL'}")
        # device / distributed
        print(f"  Device               : {self.device}")
        print(f"  Distributed          : {'YES' if self.distributed else 'NO'}")
        # config-based info (best-effort)
        dl_cfg = getattr(cfg, "DATA_LOADER", None)
        num_workers = "N/A"
        if dl_cfg is not None:
            num_workers = getattr(dl_cfg, "NUM_WORKERS", getattr(dl_cfg, "num_workers", "N/A"))
        train_cfg = getattr(cfg, "TRAIN", None)
        batch_size = getattr(train_cfg, "BATCH_SIZE", getattr(train_cfg, "batch_size", "N/A")) if train_cfg is not None else "N/A"
        print(f"  Num workers (data)   : {num_workers}")
        print(f"  Train batch size     : {batch_size}")
        print()

    def _log_section(self, title, width=70):
        """打印带分隔线的章节标题"""
        if not self.is_master:
            return
        print()
        print("=" * width)
        print(f"{title:^{width}}")
        print("=" * width)

    def _log(self, message, level="INFO", prefix=None):
        """统一的日志输出系统"""
        if not self.is_master:
            return
        
        if prefix:
            formatted_message = f"[{prefix}] {message}"
        else:
            formatted_message = f"[{level}] {message}"
        
        print(formatted_message)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning - Offline Validation')
    parser.add_argument('--folder', dest='folder', default=None, type=str)
    parser.add_argument("--resume", type=int, default=-1, help="Checkpoint epoch to load (caption_model_<N>.pth)")
    parser.add_argument("--val_samples", type=int, default=0, help="Number of validation samples to use (0 for all)")
    parser.add_argument("--corrupt_level", type=str, default="heavy", choices=["none","light","medium","heavy"],
                        help="If provided, apply corruption to image byte streams for evaluation")
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="ByteCaption-Eval", help="Wandb project name for logging.")
    parser.add_argument("--wandb_name", type=str, default=None, help="A specific name for the wandb run.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        # 尝试加载coco数据集的 config
        config_file = 'config_coco.yml'
        config_path = os.path.join(args.folder, config_file)
        if os.path.exists(config_path):
            cfg_from_file(config_path)
            print(f"[CONFIG] Loaded config: {config_path}")
        else:
            # 若找不到，仍尝试读取 generic config.yml（兼容旧项目）
            alt = os.path.join(args.folder, 'config.yml')
            if os.path.exists(alt):
                cfg_from_file(alt)
                print(f"[CONFIG] Loaded config: {alt}")
            else:
                print(f"[WARNING] Config file not found in folder: {args.folder}")

    cfg.ROOT_DIR = args.folder
    tester = Tester(args)

    # 加载策略：
    # - 如果 --resume > 0：加载 caption_model_<N>.pth
    # - 如果 --resume == -1：优先加载 best_model.pth（若存在），否则回退到自动检测最新 caption_model_*.pth
    # - 否则（resume == 0 或其他）：尝试自动检测最新 caption_model_*.pth
    if args.resume > 0:
        tester.eval(args.resume)
    elif args.resume == -1:
        best_path = os.path.join(cfg.ROOT_DIR, 'snapshot', 'best_model.pth') if cfg.ROOT_DIR else None
        if best_path and os.path.exists(best_path):
            print(f"Loading best model: {best_path}")
            tester.model.load_state_dict(torch.load(best_path, map_location=lambda s, l: s))
            tester.eval('best')
        else:
            # 回退到自动查找最新的 caption_model_*.pth
            snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot') if cfg.ROOT_DIR else None
            latest_ckpt = None
            if snapshot_folder and os.path.isdir(snapshot_folder):
                import glob, re
                files = glob.glob(os.path.join(snapshot_folder, "caption_model_*.pth"))
                if files:
                    def epoch_from_name(p):
                        m = re.search(r'caption_model_(\d+)\.pth$', p)
                        return int(m.group(1)) if m else -1
                    files.sort(key=epoch_from_name, reverse=True)
                    latest_ckpt = files[0]
            if latest_ckpt:
                import re
                m = re.search(r'caption_model_(\d+)\.pth$', latest_ckpt)
                epoch_num = int(m.group(1)) if m else -1
                print(f"Auto-detected latest checkpoint: {latest_ckpt}, epoch={epoch_num}")
                tester.model.load_state_dict(torch.load(latest_ckpt, map_location=lambda s, l: s))
                tester.eval(epoch_num)
            else:
                print("No best_model.pth and no caption_model_*.pth found. Nothing to evaluate.")
    else:
        # 自动查找最新 checkpoint（same as previous behavior）
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot') if cfg.ROOT_DIR else None
        latest_ckpt = None
        if snapshot_folder and os.path.isdir(snapshot_folder):
            import glob, re
            files = glob.glob(os.path.join(snapshot_folder, "caption_model_*.pth"))
            if files:
                def epoch_from_name(p):
                    m = re.search(r'caption_model_(\d+)\.pth$', p)
                    return int(m.group(1)) if m else -1
                files.sort(key=epoch_from_name, reverse=True)
                latest_ckpt = files[0]
        if latest_ckpt:
            import re
            m = re.search(r'caption_model_(\d+)\.pth$', latest_ckpt)
            epoch_num = int(m.group(1)) if m else -1
            print(f"Auto-detected latest checkpoint: {latest_ckpt}, epoch={epoch_num}")
            tester.model.load_state_dict(torch.load(latest_ckpt, map_location=lambda s, l: s))
            tester.eval(epoch_num)
        else:
            print("No checkpoint provided and no snapshot files found. Nothing to evaluate.")
    
    if WANDB_AVAILABLE and not args.disable_wandb:
        wandb.finish()