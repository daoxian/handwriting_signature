# coding: utf-8
import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============ 分布式与随机种子 ============

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

def set_seed(seed=42, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


# ============ 配置 ============

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')

@dataclass
class Config:
    device: str = 'cuda'
    CHISIG_DIR: str = "/path/to/ChiSig_flat_dir"
    FORGERY_DIR: str = ""  # 可选：真实模仿签名目录（仅训练用户）
    save_dir: str = "./outputs_hybrid_forge"

    # 划分
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    test_ratio: float = 0.10

    image_size: int = 224

    # 自监督预训练（可开关）
    do_pretrain: bool = True
    pretrain_epochs: int = 20
    pretrain_batch_size: int = 256
    pretrain_lr: float = 1e-3
    pretrain_temperature: float = 0.2
    pretrain_use_scheduler: bool = True
    pretrain_aug_mag: float = 1.0

    # Episodic few-shot 训练（稳定+Hybrid Forge）
    finetune_epochs: int = 80
    finetune_lr: float = 3e-4
    finetune_weight_decay: float = 1e-4
    finetune_use_scheduler: bool = True
    warmup_epochs: int = 5
    finetune_patience: int = 12
    grad_clip_norm: float = 1.0

    # episode参数：N-way, K-shot, Q-query/class
    way: int = 24
    shot: int = 6
    query: int = 8
    same_name_inject_prob: float = 0.5
    min_episodes_per_rank: int = 80

    # 原型温度（温和，避免过尖锐）
    proto_temperature: float = 0.09

    # Dropout
    dropout_p: float = 0.2

    # 验证/部署
    target_FRR: float = 0.01
    calibrate_method: str = "isotonic"
    transductive_em_iters: int = 0  # 训练期 ValAUC 不用EM
    val_kfolds: int = 5

    # 固定验证协议参数（稳定Val）
    val_impostor_per_genuine: int = 2
    val_include_same_name: bool = True
    val_use_fixed_knn: bool = True
    val_knn_k: int = 2

    # Hybrid Forge（真实+轻量合成）
    use_synthetic_forge: bool = True
    forge_per_class: int = 2
    forge_weight_max: float = 0.15  # 最终上限
    forge_warmup_epochs: int = 10
    forge_ramp_epochs: int = 10     # 从暖启动结束到达上限的线性阶段
    forge_margin: float = 0.25      # hinge margin on cosine
    forge_aug_mag: float = 0.6      # 合成增强强度（0~1，轻中等）
    forge_morph_ks: int = 3         # 形态学厚薄核大小（奇数）
    forge_knn_source_k: int = 5     # 模板近邻个数

    # 评估时是否额外生成少量合成伪造负例作为report维度（不参与阈值选择）
    eval_include_synth_forge: bool = False
    eval_synth_per_probe: int = 1

    # 并行与加载
    num_workers: int = 4


# ============ 数据与划分 ============

def list_images(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                files.append(os.path.join(dirpath, fn))
    return files

def parse_chisig_filename(path: str):
    fn = os.path.basename(path)
    base = os.path.splitext(fn)[0]
    parts = base.split('-')
    if len(parts) < 3:
        return base, None, -1
    name = '-'.join(parts[:-2])
    uid_str = parts[-2]
    try:
        index = int(parts[-1])
    except:
        index = -1
    return name, uid_str, index

def build_metadata(chisig_dir: str):
    if not os.path.isdir(chisig_dir):
        raise FileNotFoundError(f"ChiSig directory not found: {chisig_dir}")
    paths = list_images(chisig_dir)
    if len(paths) == 0:
        raise ValueError(f"No images found under {chisig_dir}")
    meta = []
    for p in paths:
        name, uid_str, index = parse_chisig_filename(p)
        if uid_str is None:
            continue
        meta.append({"path": p, "user_id_str": uid_str, "name": name, "index": index})
    if len(meta) == 0:
        raise ValueError("No valid ChiSig filenames parsed.")
    return meta

def build_forgery_metadata(forgery_dir: str):
    # 预期命名：{source_uid}-{target_uid}-*.jpg 或 {target_uid}-{source_uid}-*.jpg，尽量健壮解析
    meta = []
    if forgery_dir and os.path.isdir(forgery_dir):
        paths = list_images(forgery_dir)
        for p in paths:
            fn = os.path.basename(p)
            base = os.path.splitext(fn)[0]
            parts = base.split('-')
            if len(parts) < 2:
                continue
            # 尝试最后两个token作为 (target, source)
            target_uid = parts[-1]
            source_uid = parts[-2]
            # 如果包含非数字/非标准ID，仍然记录
            meta.append({"path": p, "target_uid": target_uid, "source_uid": source_uid})
    return meta

def writer_disjoint_split(meta: List[Dict], cfg: Config):
    users = sorted(list({m["user_id_str"] for m in meta}))
    random.shuffle(users)
    n = len(users)
    n_train = int(round(n * cfg.train_ratio))
    n_val = int(round(n * cfg.val_ratio))
    n_test = max(0, n - n_train - n_val)

    train_users = set(users[:n_train])
    val_users = set(users[n_train:n_train+n_val])
    test_users = set(users[n_train+n_val:])

    meta_train = [m for m in meta if m["user_id_str"] in train_users]
    meta_val = [m for m in meta if m["user_id_str"] in val_users]
    meta_test = [m for m in meta if m["user_id_str"] in test_users]

    if is_main_process():
        print(f"[Split] Total users: {len(users)}, Total imgs: {len(meta)}")
        print(f"  Train: {len(train_users)} users, {len(meta_train)} imgs")
        print(f"  Val:   {len(val_users)} users, {len(meta_val)} imgs")
        print(f"  Test:  {len(test_users)} users, {len(meta_test)} imgs")

    return meta_train, meta_val, meta_test


# ============ 变换 ============

def letterbox(img: Image.Image, target_size=224, bg=255):
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    new_img = Image.new("L", (target_size, target_size), color=bg)
    paste_x = (target_size - nw) // 2
    paste_y = (target_size - nh) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img

class SignatureFriendlyAugment(object):
    def __init__(self, target_size=224, magnitude=0.6):
        m = float(max(0.0, min(1.0, magnitude)))
        self.target_size = target_size
        degrees = 10 * m
        translate = 0.08 * m
        scale_low = 1.0 - 0.08 * m
        scale_high = 1.0 + 0.08 * m
        shear = 5 * m
        persp = 0.08 * m
        blur_sigma = (0.05, 1.0 * m + 0.05)
        self.affine = transforms.RandomAffine(
            degrees=degrees, translate=(translate, translate),
            scale=(scale_low, scale_high), shear=shear,
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.persp = transforms.RandomPerspective(distortion_scale=persp, p=0.5,
                                                  interpolation=transforms.InterpolationMode.BILINEAR)
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=blur_sigma)
        self.to_tensor = transforms.ToTensor()
    def __call__(self, img: Image.Image):
        img = img.convert('L')
        img = letterbox(img, target_size=self.target_size, bg=255)
        img = self.affine(img)
        img = self.persp(img)
        img = self.blur(img)
        t = self.to_tensor(img)
        t = (t - 0.5) / 0.5
        return t

class SignatureDeterministicTransform(object):
    def __init__(self, target_size=224):
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()
    def __call__(self, img: Image.Image):
        img = img.convert('L')
        img = letterbox(img, target_size=self.target_size, bg=255)
        t = self.to_tensor(img)
        t = (t - 0.5) / 0.5
        return t

# 轻量伪造增强（贴近真实模仿）
class ImpostorForgeAugment(object):
    def __init__(self, target_size=224, magnitude=0.6, morph_ks=3):
        m = float(max(0.0, min(1.0, magnitude)))
        self.target_size = target_size
        degrees = 8 * m
        translate = 0.06 * m
        scale_low = 1.0 - 0.06 * m
        scale_high = 1.0 + 0.06 * m
        shear = 4 * m
        persp = 0.06 * m
        blur_sigma = (0.05, 0.8 * m + 0.05)
        self.affine = transforms.RandomAffine(
            degrees=degrees, translate=(translate, translate),
            scale=(scale_low, scale_high), shear=shear,
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.persp = transforms.RandomPerspective(distortion_scale=persp, p=0.5,
                                                  interpolation=transforms.InterpolationMode.BILINEAR)
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=blur_sigma)
        self.to_tensor = transforms.ToTensor()
        self.morph_ks = int(max(1, morph_ks))
        if self.morph_ks % 2 == 0:
            self.morph_ks += 1
    def _morph_thick_thin(self, t01: torch.Tensor, thicken=True):
        # 白底黑字：墨水=1 - t01
        ink = 1.0 - t01
        op = F.max_pool2d if thicken else F.avg_pool2d
        k = self.morph_ks
        pad = k // 2
        ink2 = op(ink.unsqueeze(0), kernel_size=k, stride=1, padding=pad).squeeze(0)
        return torch.clamp(1.0 - ink2, 0.0, 1.0)
    def __call__(self, img: Image.Image):
        img = img.convert('L')
        img = letterbox(img, target_size=self.target_size, bg=255)
        img = self.affine(img)
        img = self.persp(img)
        t = self.to_tensor(img)
        t = self.blur(transforms.ToPILImage()(t))
        t01 = self.to_tensor(t)
        if random.random() < 0.5:
            t01 = self._morph_thick_thin(t01, thicken=(random.random() < 0.5))
        return (t01 - 0.5) / 0.5


# ============ 数据集与采样器 ============

class LabeledSignatureDatasetFromMeta(Dataset):
    def __init__(self, meta_list: List[Dict], transform, target_size=224):
        assert len(meta_list) > 0
        self.meta = meta_list
        self.transform = transform
        uid_strs = sorted(list({m["user_id_str"] for m in self.meta}))
        self.uid_map = {u: i for i, u in enumerate(uid_strs)}
        self.uid_int_to_str = {i: u for u, i in self.uid_map.items()}
        self.samples = []
        self.uid_to_name: Dict[int, str] = {}
        self.name_to_uids: Dict[str, List[int]] = {}
        for m in self.meta:
            uid_int = self.uid_map[m["user_id_str"]]
            self.samples.append((m["path"], uid_int, m["name"], m["user_id_str"], m["index"]))
            self.uid_to_name[uid_int] = m["name"]
        for uid_int, name in self.uid_to_name.items():
            self.name_to_uids.setdefault(name, []).append(uid_int)
        self.user_indices: Dict[int, List[int]] = {}
        for idx, (_, uid_int, _, _, _) in enumerate(self.samples):
            self.user_indices.setdefault(uid_int, []).append(idx)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, uid_int, name, uid_str, index = self.samples[idx]
        img = Image.open(path).convert('L')
        x = self.transform(img)
        return x, uid_int, path, uid_str, index

class DistributedEpisodicSampler(Sampler):
    def __init__(self, dataset: LabeledSignatureDatasetFromMeta, way=16, shot=6, query=6,
                 same_name_inject_prob=0.5, world_size=1, rank=0, seed=42, min_episodes_per_rank=60):
        self.ds = dataset
        self.way = way
        self.shot = shot
        self.query = query
        self.same_name_inject_prob = same_name_inject_prob
        self.world_size = max(1, world_size)
        self.rank = rank
        self.seed = seed
        self.min_episodes_per_rank = max(1, int(min_episodes_per_rank))
        self.user_ids = list(self.ds.user_indices.keys())
        self.name_groups = [uids for uids in self.ds.name_to_uids.values() if len(uids) >= 2]
        self.epoch = 0
        self._episodes_this_rank: List[List[int]] = []

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._episodes_this_rank = self._build_epoch_episodes(epoch)

    def _choose_users_for_episode(self, g: random.Random) -> List[int]:
        selected = set()
        if len(self.name_groups) > 0 and g.random() < self.same_name_inject_prob:
            group = g.choice(self.name_groups)
            if len(group) >= 2:
                pair = g.sample(group, 2)
                selected.update(pair)
        user_ids_shuf = self.user_ids.copy()
        g.shuffle(user_ids_shuf)
        for uid in user_ids_shuf:
            if len(selected) >= self.way:
                break
            selected.add(uid)
        if len(selected) < self.way:
            while len(selected) < self.way:
                selected.add(g.choice(self.user_ids))
        return list(selected)[:self.way]

    def _build_one_episode_indices(self, g: random.Random, users: List[int]) -> List[int]:
        idxs = []
        KQ = self.shot + self.query
        for uid in users:
            pool = self.ds.user_indices[uid]
            chosen = [pool[g.randrange(len(pool))] for _ in range(KQ)]
            idxs.extend(chosen)
        return idxs

    def _build_epoch_episodes(self, epoch: int) -> List[List[int]]:
        g = random.Random(self.seed + epoch)
        episodes = []
        target_total = self.world_size * self.min_episodes_per_rank
        while len(episodes) < target_total:
            users = self._choose_users_for_episode(g)
            ep = self._build_one_episode_indices(g, users)
            episodes.append(ep)
        per_rank = max(1, len(episodes) // self.world_size)
        usable = per_rank * self.world_size
        episodes = episodes[:usable]
        my_eps = episodes[self.rank * per_rank : (self.rank + 1) * per_rank]
        return my_eps

    def __iter__(self):
        if not self._episodes_this_rank:
            self.set_epoch(self.epoch)
        for ep in self._episodes_this_rank:
            yield ep

    def __len__(self):
        return len(self._episodes_this_rank)


# ============ 模型与损失 ============

class SignatureEncoder(nn.Module):
    def __init__(self, emb_dim=256, dropout_p=0.2):
        super().__init__()
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(512, emb_dim)
    def forward(self, x):
        h = self.backbone(x)
        h = torch.flatten(h, 1)
        h = self.dropout(h)
        z = self.fc(h)
        z = F.normalize(z, dim=-1)  # 球面嵌入
        return z

def prototypical_logits(emb: torch.Tensor, way: int, shot: int, query: int, temperature: float):
    B, D = emb.shape
    assert B == way * (shot + query), f"Batch size {B} != way*(shot+query)={way*(shot+query)}"
    emb = emb.view(way, shot + query, D)
    support = emb[:, :shot, :]
    query_emb = emb[:, shot:, :].contiguous().view(way * query, D)
    proto = support.mean(dim=1)
    proto = F.normalize(proto, dim=-1)
    logits = torch.matmul(query_emb, proto.T) / max(1e-6, temperature)
    target = torch.arange(way, device=emb.device).repeat_interleave(query)
    return logits, target, proto, query_emb

def prototypical_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)

def forge_hinge_loss(imp_feats: torch.Tensor, imp_targets: torch.Tensor, proto: torch.Tensor, margin: float):
    # imp_feats: [Nf,D] (normalized), proto: [way,D] (normalized)
    if imp_feats is None or imp_feats.shape[0] == 0:
        return torch.tensor(0.0, device=proto.device)
    proto_sel = proto[imp_targets]  # [Nf,D]
    cos = torch.sum(imp_feats * proto_sel, dim=-1)  # cosine similarity in [-1,1]
    loss = torch.relu(cos - margin).mean()
    return loss


# ============ 自监督预训练 ============

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=256, proj_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, proj_dim),
        )
    def forward(self, x):
        return self.net(x)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.t = temperature
    def forward(self, p1, p2):
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        z = torch.cat([p1, p2], dim=0)
        sim = torch.matmul(z, z.T) / self.t
        N = z.shape[0]
        B = p1.shape[0]
        mask = torch.eye(N, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        targets = torch.arange(N, device=z.device)
        targets = torch.where(targets < B, targets + B, targets - B)
        loss = F.cross_entropy(sim, targets)
        return loss

def pretrain_simclr_ddp(unlabeled_paths: List[str], cfg: Config, rank: int, world_size: int, local_rank: int) -> nn.Module:
    device = torch.device('cuda', local_rank)

    class UnlabeledTwoViewsDataset(Dataset):
        def __init__(self, paths, t1, t2):
            self.paths = paths
            self.t1 = t1; self.t2 = t2
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('L')
            return self.t1(img), self.t2(img)

    t1 = SignatureFriendlyAugment(target_size=cfg.image_size, magnitude=cfg.pretrain_aug_mag)
    t2 = SignatureFriendlyAugment(target_size=cfg.image_size, magnitude=cfg.pretrain_aug_mag)
    ds = UnlabeledTwoViewsDataset(unlabeled_paths, t1, t2)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    batch_size = max(1, cfg.pretrain_batch_size // world_size)
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                    num_workers=cfg.num_workers, pin_memory=True, drop_last=True, persistent_workers=(cfg.num_workers>0))

    encoder = SignatureEncoder(emb_dim=256, dropout_p=cfg.dropout_p).to(device)
    proj = ProjectionHead(in_dim=256, proj_dim=128).to(device)

    encoder = nn.parallel.DistributedDataParallel(encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    proj = nn.parallel.DistributedDataParallel(proj, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=cfg.pretrain_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.pretrain_epochs) if cfg.pretrain_use_scheduler else None
    criterion = NTXentLoss(temperature=cfg.pretrain_temperature)

    for epoch in range(cfg.pretrain_epochs):
        sampler.set_epoch(epoch)
        encoder.train(); proj.train()
        total_loss = 0.0; steps = 0
        for v1, v2 in dl:
            v1 = v1.to(device); v2 = v2.to(device)
            x = torch.cat([v1, v2], dim=0)
            z = encoder(x)
            p = proj(z)
            B = v1.size(0)
            p1 = p[:B]; p2 = p[B:]
            loss = criterion(p1, p2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); steps += 1
        loss_tensor = torch.tensor([total_loss, steps], dtype=torch.float32, device=device)
        if dist.is_initialized():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_all = (loss_tensor[0] / (loss_tensor[1] + 1e-9)).item()
        if is_main_process():
            print(f"[Pretrain][Epoch {epoch+1}/{cfg.pretrain_epochs}] Loss={avg_loss_all:.4f}")
        if scheduler is not None and steps > 0:
            scheduler.step()

    return encoder.module


# ============ EMA ============

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            t = v.detach().clone()
            t.requires_grad = False
            self.shadow[k] = t
    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            if (k not in self.shadow) or (self.shadow[k].dtype != v.dtype) or (self.shadow[k].shape != v.shape):
                t = v.detach().clone()
                t.requires_grad = False
                self.shadow[k] = t
                continue
            if self.shadow[k].device != v.device:
                self.shadow[k] = self.shadow[k].to(v.device)
            if torch.is_floating_point(v):
                self.shadow[k].mul_((self.decay)).add_(v.detach(), alpha=(1.0 - self.decay))
            else:
                self.shadow[k].copy_(v.detach())
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


# ============ 原型、评分、固定验证协议 ============

def compute_feature(encoder: nn.Module, path: str, cfg: Config) -> np.ndarray:
    device = torch.device(cfg.device)
    img = Image.open(path).convert('L')
    x = SignatureDeterministicTransform(cfg.image_size)(img).unsqueeze(0).to(device)
    encoder.eval()
    with torch.no_grad():
        feat = encoder(x).cpu().numpy()[0]  
    return feat

def build_prototypes_cosine(encoder: nn.Module, meta_list: List[Dict], k_enroll: int, cfg: Config):
    uid_groups: Dict[str, List[Dict]] = {}
    for m in meta_list:
        uid_groups.setdefault(m["user_id_str"], []).append(m)
    prototypes: Dict[str, np.ndarray] = {}
    enroll_map: Dict[str, List[Dict]] = {}
    probe_meta: List[Dict] = []
    for uid, lst in uid_groups.items():
        lst_sorted = sorted(lst, key=lambda x: ((x["index"] if x["index"] is not None and x["index"] >= 0 else 10**9), x["path"]))
        regs = lst_sorted[:k_enroll]
        enroll_map[uid] = regs
        probe_meta.extend(lst_sorted[k_enroll:])
    for uid, regs in enroll_map.items():
        feats = [compute_feature(encoder, m["path"], cfg) for m in regs]
        mu = np.mean(np.stack(feats, axis=0), axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        prototypes[uid] = mu.astype(np.float32)
    return prototypes, enroll_map, probe_meta

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def build_fixed_val_protocol(meta_val: List[Dict], cfg: Config, enroll_k: int,
                             base_encoder_for_knn: Optional[nn.Module] = None) -> Dict:
    # 返回 claims: List[(path, claimed_uid, label, tag)]，其中 tag ∈ {'genuine','rand','same_name','knn'}
    uid_groups: Dict[str, List[Dict]] = {}
    name_to_uids: Dict[str, List[str]] = {}
    for m in meta_val:
        uid_groups.setdefault(m["user_id_str"], []).append(m)
        name_to_uids.setdefault(m["name"], []).append(m["user_id_str"])
    claims_fixed = []
    rng = random.Random(12345)
    prototypes_base = None
    if cfg.val_use_fixed_knn and base_encoder_for_knn is not None:
        prototypes_base, enroll_map_base, probe_meta_base = build_prototypes_cosine(base_encoder_for_knn, meta_val, k_enroll=enroll_k, cfg=cfg)
        q_feat_cache: Dict[str, np.ndarray] = {}
        for m in probe_meta_base:
            q_feat_cache[m["path"]] = compute_feature(base_encoder_for_knn, m["path"], cfg)

    all_uids = sorted(uid_groups.keys())
    for uid, lst in uid_groups.items():
        lst_sorted = sorted(lst, key=lambda x: ((x["index"] if x["index"] is not None and x["index"] >= 0 else 10**9), x["path"]))
        regs = lst_sorted[:enroll_k]
        probes = lst_sorted[enroll_k:]
        for pm in probes:
            path = pm["path"]
            # genuine
            claims_fixed.append((path, uid, 1, 'genuine'))
            # 随机 impostor
            neg_uids = [u for u in all_uids if u != uid]
            rng.shuffle(neg_uids)
            for j in range(cfg.val_impostor_per_genuine):
                if j < len(neg_uids):
                    claims_fixed.append((path, neg_uids[j], 0, 'rand'))
            # 同名 impostor
            if cfg.val_include_same_name:
                same_uids = [u for u in set(name_to_uids.get(pm["name"], [])) if u != uid]
                if len(same_uids) > 0:
                    claims_fixed.append((path, rng.choice(same_uids), 0, 'same_name'))
            # kNN impostor（基于初始编码器，固定一次）
            if cfg.val_use_fixed_knn and base_encoder_for_knn is not None and prototypes_base is not None:
                qf = q_feat_cache.get(path, None)
                if qf is not None:
                    sims = [(other_uid, cosine_similarity(qf, prototypes_base[other_uid])) for other_uid in prototypes_base.keys() if other_uid != uid]
                    sims.sort(key=lambda x: x[1], reverse=True)
                    for j in range(min(cfg.val_knn_k, len(sims))):
                        claims_fixed.append((path, sims[j][0], 0, 'knn'))
    return {"claims": claims_fixed, "enroll_k": enroll_k}

def score_claims(encoder: nn.Module, claims: List[Tuple[str, str, int, str]],
                 meta_val: List[Dict], enroll_k: int, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prototypes, enroll_map, _ = build_prototypes_cosine(encoder, meta_val, k_enroll=enroll_k, cfg=cfg)
    scores, labels, tags = [], [], []
    for path, claimed_uid, label, tag in claims:
        if claimed_uid not in prototypes:
            continue
        q = compute_feature(encoder, path, cfg)
        s = cosine_similarity(q, prototypes[claimed_uid])
        scores.append(s); labels.append(label); tags.append(tag)
    return np.array(scores), np.array(labels), np.array(tags)

def evaluate_AUC(scores: np.ndarray, labels: np.ndarray):
    if SKLEARN_AVAILABLE and len(scores) > 0 and len(labels) > 0:
        try:
            return roc_auc_score(labels, scores)
        except Exception:
            return None
    return None

class Calibrator:
    def __init__(self, method="isotonic"):
        self.method = method
        self._iso = None
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        if self.method == "isotonic" and SKLEARN_AVAILABLE and len(scores) > 0:
            self._iso = IsotonicRegression(out_of_bounds='clip')
            self._iso.fit(scores, labels)
        else:
            self._iso = None
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if self._iso is not None:
            return self._iso.transform(scores)
        z = scores
        p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        return p

def find_threshold_at_FRR(scores: np.ndarray, labels: np.ndarray, target_FRR=0.01):
    genuine_scores = scores[labels == 1]
    n = genuine_scores.shape[0]
    if n == 0:
        raise ValueError("No genuine samples in validation.")
    k = int(math.floor(target_FRR * max(1, n - 1)))
    k = max(0, min(k, n - 1))
    kth = np.partition(genuine_scores, k)[k]
    return float(kth)

def compute_FAR_FRR(scores: np.ndarray, labels: np.ndarray, thr: float):
    preds = (scores >= thr).astype(np.int32)
    genuine = (labels == 1)
    impostor = (labels == 0)
    FRR = float(np.sum((preds == 0) & genuine)) / max(1, np.sum(genuine))
    FAR = float(np.sum((preds == 1) & impostor)) / max(1, np.sum(impostor))
    return FAR, FRR

def subset_metrics(scores_cal: np.ndarray, labels: np.ndarray, tags: np.ndarray, thr: float):
    # 整体
    FAR_all, FRR_all = compute_FAR_FRR(scores_cal, labels, thr)
    AUC_all = evaluate_AUC(scores_cal, labels)
    overall = {"FAR": FAR_all, "FRR": FRR_all, "AUC": AUC_all}

    # 子集计算
    def filt(tag_name: str):
        idx = (tags == tag_name)
        if not np.any(idx):
            return {"FAR": None, "FRR": None, "AUC": None}
        FAR_sub, FRR_sub = compute_FAR_FRR(scores_cal[idx], labels[idx], thr)
        AUC_sub = evaluate_AUC(scores_cal[idx], labels[idx])
        return {"FAR": FAR_sub, "FRR": FRR_sub, "AUC": AUC_sub}

    same_name = filt('same_name')
    knn = filt('knn')
    return overall, same_name, knn


# ============ 训练（Warmup+Cosine、EMA、固定Val协议、混合伪造） ============

def create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def precompute_knn_sources(train_ds: LabeledSignatureDatasetFromMeta, encoder: nn.Module, cfg: Config, device: torch.device):
    # 用训练集每类原型做kNN，返回：uid_int -> [近邻uid_int...]
    train_meta = [{"path": p, "user_id_str": uid_str, "name": name, "index": idx}
                  for (p, uid_int, name, uid_str, idx) in train_ds.samples]
    prototypes, _, _ = build_prototypes_cosine(encoder, train_meta, k_enroll=4, cfg=cfg)
    # 按train_ds的uid映射转为int索引
    uids_str = list(prototypes.keys())
    proto_mat = []
    proto_ints = []
    for us in uids_str:
        if us in train_ds.uid_map:
            proto_mat.append(prototypes[us])
            proto_ints.append(train_ds.uid_map[us])
    if len(proto_mat) == 0:
        return {}
    P = torch.from_numpy(np.stack(proto_mat, axis=0)).to(device)  # [U,D]
    sims = torch.matmul(P, P.T)
    sims.fill_diagonal_(-1.0)
    knn_map = {}
    for i, uid_int in enumerate(proto_ints):
        _, topk = torch.topk(sims[i], k=min(cfg.forge_knn_source_k, sims.shape[1]-1))
        knn_map[uid_int] = [proto_ints[j] for j in topk.tolist()]
    return knn_map

def build_real_forgery_map(forgery_meta: List[Dict], train_users: set):
    m = {}
    for f in forgery_meta:
        tgt = f["target_uid"]
        if tgt in train_users:
            m.setdefault(tgt, []).append(f["path"])
    return m

def finetune_hybrid_forge_ddp(student: nn.Module,
                              train_ds: LabeledSignatureDatasetFromMeta,
                              meta_val: List[Dict],
                              cfg: Config,
                              fixed_val_proto: Dict,
                              real_forgery_map: Dict[str, List[str]],
                              knn_source_map: Dict[int, List[int]],
                              rank: int, world_size: int, local_rank: int) -> nn.Module:
    device = torch.device('cuda', local_rank)

    student = student.to(device)
    student_ddp = nn.parallel.DistributedDataParallel(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(student_ddp.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.finetune_weight_decay)
    scheduler = create_warmup_cosine_scheduler(optimizer, cfg.warmup_epochs, cfg.finetune_epochs) if cfg.finetune_use_scheduler else None

    sampler = DistributedEpisodicSampler(train_ds, way=cfg.way, shot=cfg.shot, query=cfg.query,
                                         same_name_inject_prob=cfg.same_name_inject_prob,
                                         world_size=world_size, rank=rank, seed=42,
                                         min_episodes_per_rank=cfg.min_episodes_per_rank)
    dl = DataLoader(train_ds, batch_sampler=sampler,
                    num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers>0))

    ema = EMA(student_ddp.module, decay=0.999)
    forge_aug = ImpostorForgeAugment(target_size=cfg.image_size, magnitude=cfg.forge_aug_mag, morph_ks=cfg.forge_morph_ks)

    best_auc_smooth = -1.0
    best_state = None
    no_improve = 0
    auc_hist = []

    for epoch in range(cfg.finetune_epochs):
        sampler.set_epoch(epoch)
        student_ddp.train()
        total_loss = 0.0; steps = 0

        # forge权重分段增益
        if epoch < cfg.forge_warmup_epochs:
            forge_w = 0.0
        elif epoch < (cfg.forge_warmup_epochs + cfg.forge_ramp_epochs):
            frac = float(epoch - cfg.forge_warmup_epochs + 1) / float(cfg.forge_ramp_epochs)
            forge_w = min(cfg.forge_weight_max, cfg.forge_weight_max * frac)
        else:
            forge_w = cfg.forge_weight_max

        for batch in dl:
            xs, uid_ints, paths, uid_strs, idxs = batch
            xs = xs.to(device)
            uid_ints = uid_ints.to(device)

            # 学生前向与原型loss
            feat = student_ddp(xs)  # [B,D]
            logits, target, proto, query_emb = prototypical_logits(feat, cfg.way, cfg.shot, cfg.query, cfg.proto_temperature)
            loss_proto = prototypical_loss_from_logits(logits, target)

            # 合成/真实伪造负例（训练期）
            loss_forge = torch.tensor(0.0, device=device)
            if cfg.use_synthetic_forge and cfg.forge_per_class > 0 and forge_w > 0:
                uid_w = uid_ints.view(cfg.way, cfg.shot + cfg.query)
                uid_episode_ints = uid_w[:, 0]  # 每类第一个样本的uid_int
                imp_feats = []
                imp_targets = []
                for j in range(cfg.way):
                    uid_int = uid_episode_ints[j].item()
                    uid_str = train_ds.uid_int_to_str[uid_int]
                    # 先尝试真实伪造
                    real_pool = real_forgery_map.get(uid_str, [])
                    used = 0
                    # 真实伪造
                    while used < cfg.forge_per_class and len(real_pool) > 0:
                        src_path = random.choice(real_pool)
                        try:
                            img = Image.open(src_path).convert('L')
                        except Exception:
                            break
                        x_imp = SignatureDeterministicTransform(cfg.image_size)(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            z_imp = student_ddp.module(x_imp)  # [1,D]
                        imp_feats.append(z_imp.squeeze(0))
                        imp_targets.append(j)
                        used += 1
                    # 不足则补合成伪造
                    while used < cfg.forge_per_class:
                        source_candidates = knn_source_map.get(uid_int, [])
                        if len(source_candidates) == 0:
                            source_candidates = [u for u in train_ds.user_indices.keys() if u != uid_int]
                        if len(source_candidates) == 0: break
                        src_uid = random.choice(source_candidates)
                        src_pool_idx = train_ds.user_indices[src_uid]
                        src_idx = random.choice(src_pool_idx)
                        src_path = train_ds.samples[src_idx][0]
                        try:
                            img = Image.open(src_path).convert('L')
                        except Exception:
                            break
                        x_imp = forge_aug(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            z_imp = student_ddp.module(x_imp)
                        imp_feats.append(z_imp.squeeze(0))
                        imp_targets.append(j)
                        used += 1
                if len(imp_feats) > 0:
                    imp_feats_t = torch.stack(imp_feats, dim=0)  # [Nf,D]
                    imp_targets_t = torch.tensor(imp_targets, dtype=torch.long, device=device)
                    loss_forge = forge_hinge_loss(imp_feats_t, imp_targets_t, proto, margin=cfg.forge_margin)

            loss = loss_proto + forge_w * loss_forge

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(student_ddp.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            ema.update(student_ddp.module)

            total_loss += loss.item(); steps += 1

        if scheduler is not None and steps > 0:
            scheduler.step()

        # 验证（使用EMA模型 + 固定Val协议）
        metric_tensor = torch.tensor([-1.0], dtype=torch.float32, device=device)
        if is_main_process():
            tmp_model = SignatureEncoder(emb_dim=256, dropout_p=cfg.dropout_p).to(device)
            ema.copy_to(tmp_model)
            claims = fixed_val_proto["claims"]
            enroll_k = fixed_val_proto["enroll_k"]
            scores, labels, tags = score_claims(tmp_model, claims, meta_val, enroll_k, cfg)
            val_auc = evaluate_AUC(scores, labels) or 0.0

            auc_hist.append(val_auc)
            window = 5
            val_auc_smooth = float(np.mean(auc_hist[-window:])) if len(auc_hist) >= window else float(np.mean(auc_hist))
            metric_tensor[0] = val_auc_smooth

            improved = val_auc_smooth > best_auc_smooth
            if improved:
                best_auc_smooth = val_auc_smooth
                best_state = {k: v.clone() for k, v in ema.shadow.items()}  # 保存EMA权重
                no_improve = 0
                os.makedirs(cfg.save_dir, exist_ok=True)
                torch.save(best_state, os.path.join(cfg.save_dir, "encoder_hybrid_forge_best_ema.pth"))
            else:
                no_improve += 1

            lr = optimizer.param_groups[0]['lr']
            print(f"[HybridForge][Epoch {epoch+1}/{cfg.finetune_epochs}] lr={lr:.6e} loss={total_loss/max(1,steps):.4f} "
                  f"ValAUC={val_auc:.4f} ValAUC_MA={val_auc_smooth:.4f} BestMA={best_auc_smooth:.4f} Pat={no_improve}/{cfg.finetune_patience}"
                  f"{'  <-- New best' if improved else ''}")

        if dist.is_initialized():
            dist.broadcast(metric_tensor, src=0)

        if is_main_process() and no_improve >= cfg.finetune_patience:
            break

    enc_best = SignatureEncoder(emb_dim=256, dropout_p=cfg.dropout_p).to(torch.device('cuda'))
    if is_main_process() and best_state is not None:
        enc_best.load_state_dict(best_state)
    return enc_best


# ============ K折阈值与最终评估 ============

def kfold_val_threshold_and_calibrator(encoder: nn.Module, meta_val: List[Dict], cfg: Config, enroll_k: int):
    users = sorted(list({m["user_id_str"] for m in meta_val}))
    random.shuffle(users)
    K = max(2, cfg.val_kfolds)
    folds = [users[i::K] for i in range(K)]
    fold_thresholds = []
    for ki in range(K):
        train_users_folds = set([u for j, fold in enumerate(folds) if j != ki for u in fold])
        meta_trainfold = [m for m in meta_val if m["user_id_str"] in train_users_folds]
        if len(meta_trainfold) == 0:
            continue
        base_proto = build_fixed_val_protocol(meta_trainfold, cfg, enroll_k, encoder if cfg.val_use_fixed_knn else None)
        scores_tr, labels_tr, tags_tr = score_claims(encoder, base_proto["claims"], meta_trainfold, enroll_k, cfg)
        calib_fold = Calibrator(method=cfg.calibrate_method)
        calib_fold.fit(scores_tr, labels_tr)
        scores_tr_cal = calib_fold.predict_proba(scores_tr)
        thr_fold = find_threshold_at_FRR(scores_tr_cal, labels_tr, target_FRR=cfg.target_FRR)
        fold_thresholds.append(thr_fold)
    robust_thr = float(np.median(fold_thresholds)) if len(fold_thresholds) > 0 else None
    # 全Val拟合最终校准器
    base_proto_all = build_fixed_val_protocol(meta_val, cfg, enroll_k, encoder if cfg.val_use_fixed_knn else None)
    scores_all, labels_all, tags_all = score_claims(encoder, base_proto_all["claims"], meta_val, enroll_k, cfg)
    calib_all = Calibrator(method=cfg.calibrate_method)
    calib_all.fit(scores_all, labels_all)
    return robust_thr, calib_all

def final_eval_with_subsets(encoder: nn.Module, meta_split: List[Dict], cfg: Config, enroll_k: int,
                            calibrator: Calibrator, thr_use: Optional[float], split_name="Val"):
    base_proto = build_fixed_val_protocol(meta_split, cfg, enroll_k, encoder if cfg.val_use_fixed_knn else None)
    scores, labels, tags = score_claims(encoder, base_proto["claims"], meta_split, enroll_k, cfg)
    scores_cal = calibrator.predict_proba(scores)
    if thr_use is None:
        thr_use = find_threshold_at_FRR(scores_cal, labels, target_FRR=cfg.target_FRR)

    overall, same_name, knn = subset_metrics(scores_cal, labels, tags, thr_use)

    print(f"[{split_name} K={enroll_k}] thr={thr_use:.6f} "
          f"FAR={overall['FAR']*100:.2f}% FRR={overall['FRR']*100:.2f}% "
          f"AUC={overall['AUC'] if overall['AUC'] is not None else 'N/A'}")

    if same_name["AUC"] is not None:
        print(f"  └ same_name: FAR={same_name['FAR']*100:.2f}% FRR={same_name['FRR']*100:.2f}% AUC={same_name['AUC']:.4f}")
    if knn["AUC"] is not None:
        print(f"  └ knn_top:   FAR={knn['FAR']*100:.2f}% FRR={knn['FRR']*100:.2f}% AUC={knn['AUC']:.4f}")

    return thr_use

# ============ 主流程 ============

def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--chisig_dir', type=str, required=True, help='Path to ChiSig flat directory')
    ap.add_argument('--save_dir', type=str, default='./outputs_hybrid_forge', help='Output dir')
    ap.add_argument('--forgery_dir', type=str, default='', help='Optional real forgery dir')
    ap.add_argument('--no_pretrain', action='store_true', help='Disable SimCLR pretraining')
    ap.add_argument('--way', type=int, default=None)
    ap.add_argument('--shot', type=int, default=None)
    ap.add_argument('--query', type=int, default=None)
    ap.add_argument('--min_episodes_per_rank', type=int, default=None)
    return ap.parse_args()

def main():
    rank, world_size, local_rank = setup_distributed()
    set_seed(42, rank)
    device = torch.device('cuda', local_rank)

    args = parse_args()
    cfg = Config()
    cfg.device = 'cuda'
    cfg.CHISIG_DIR = args.chisig_dir
    cfg.FORGERY_DIR = args.forgery_dir
    cfg.save_dir = args.save_dir
    cfg.do_pretrain = not args.no_pretrain
    if args.way is not None: cfg.way = args.way
    if args.shot is not None: cfg.shot = args.shot
    if args.query is not None: cfg.query = args.query
    if args.min_episodes_per_rank is not None: cfg.min_episodes_per_rank = args.min_episodes_per_rank

    if is_main_process():
        os.makedirs(cfg.save_dir, exist_ok=True)

    # 构建与划分
    if is_main_process():
        print("==== Stage 0: Build ChiSig metadata and writer-disjoint split ====")
    meta_all = build_metadata(cfg.CHISIG_DIR)
    meta_train, meta_val, meta_test = writer_disjoint_split(meta_all, cfg)
    train_user_set = set({m["user_id_str"] for m in meta_train})

    # 自监督预训练（可选）
    if is_main_process():
        print("==== Stage 1: Self-supervised pretraining (SimCLR, DDP) ====")
    if cfg.do_pretrain:
        unlabeled_paths = [m["path"] for m in meta_all]
        base_encoder = pretrain_simclr_ddp(unlabeled_paths, cfg, rank, world_size, local_rank)
    else:
        base_encoder = SignatureEncoder(emb_dim=256, dropout_p=cfg.dropout_p).to(device)

    # 训练数据集
    train_transform = SignatureFriendlyAugment(target_size=cfg.image_size, magnitude=0.6)
    train_ds = LabeledSignatureDatasetFromMeta(meta_train, transform=train_transform, target_size=cfg.image_size)

    # 固定 Val 协议（用预训练后的base_encoder构建一次knn难负并固定）
    if is_main_process():
        print("==== Stage 2: Build fixed Val protocol (stable evaluation) ====")
    fixed_val_proto = build_fixed_val_protocol(meta_val, cfg, enroll_k=1, base_encoder_for_knn=base_encoder if cfg.val_use_fixed_knn else None)
    if is_main_process():
        claims = fixed_val_proto["claims"]
        n_claims = len(claims)
        n_pos = sum(1 for _, _, lab, _ in claims if lab == 1)
        n_neg = n_claims - n_pos
        n_users = len({m["user_id_str"] for m in meta_val})
        uid_groups = {}
        for m in meta_val:
            uid_groups.setdefault(m["user_id_str"], []).append(m)
        n_probes = sum(max(0, len(lst) - fixed_val_proto["enroll_k"]) for lst in uid_groups.values())
        print(f"[Stage2] Val users={n_users}, probes={n_probes}, claims={n_claims}, pos={n_pos}, neg={n_neg}, same_name={cfg.val_include_same_name}, fixed_knn={cfg.val_use_fixed_knn}")

    # 真实伪造映射（仅训练用户）
    forgery_meta = build_forgery_metadata(cfg.FORGERY_DIR) if cfg.FORGERY_DIR else []
    real_forgery_map = build_real_forgery_map(forgery_meta, train_user_set)

    # 训练集kNN模板近邻预计算
    if is_main_process():
        print("==== Stage 3: Precompute kNN source templates for synthetic forge ====")
    knn_source_map = precompute_knn_sources(train_ds, base_encoder, cfg, device)

    # 监督few-shot训练（混合伪造）
    if is_main_process():
        print("==== Stage 4: Episodic few-shot training (Hybrid Forge: Proto + Hinge forge, EMA eval, fixed Val) ====")
    encoder_best_ema = finetune_hybrid_forge_ddp(base_encoder, train_ds, meta_val, cfg, fixed_val_proto,
                                                 real_forgery_map, knn_source_map, rank, world_size, local_rank)

    # 保存模型
    if is_main_process():
        enc_path = os.path.join(cfg.save_dir, "encoder_hybrid_forge_best_ema_final.pth")
        torch.save(encoder_best_ema.state_dict(), enc_path)
        print(f"[Save] EMA encoder saved to {enc_path}")

    if dist.is_initialized():
        dist.barrier()

    # 最终评估
    if is_main_process():
        print("==== Stage 5: K-fold thresholding on Val & Final Evaluation (EMA model) ====")
        for enroll_k in [1, 2]:
            robust_thr, calibrator = kfold_val_threshold_and_calibrator(encoder_best_ema, meta_val, cfg, enroll_k)
            thr_info = f"(robust median thr={robust_thr:.6f})" if robust_thr is not None else "(robust thr N/A)"
            print(f"[Val K={enroll_k}] Robust threshold {thr_info}")
            thr_use = final_eval_with_subsets(encoder_best_ema, meta_val, cfg, enroll_k, calibrator, robust_thr, split_name="Val")
            print("==== Stage 6: Testing (apply Val threshold and calibration) ====")
            _ = final_eval_with_subsets(encoder_best_ema, meta_test, cfg, enroll_k, calibrator, thr_use, split_name="Test")

        print("==== Done ====")

    if dist.is_initialized():
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
