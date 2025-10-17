import os
import json
import argparse
import numpy as np
from typing import List, Dict, Optional

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ============ 模型与变换 ============

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

class SignatureDeterministicTransform(object):
    def __init__(self, target_size=224):
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()
    def __call__(self, img: Image.Image):
        img = img.convert('L')
        img = letterbox(img, target_size=self.target_size, bg=255)
        t = self.to_tensor(img)      # [1,H,W], 0~1
        t = (t - 0.5) / 0.5          # [-1,1]
        return t

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
        z = F.normalize(z, dim=-1)  # 单位球归一化
        return z

def load_encoder(model_path: str, device: torch.device, emb_dim=256, dropout_p=0.2):
    model = SignatureEncoder(emb_dim=emb_dim, dropout_p=dropout_p).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def compute_feature(encoder: nn.Module, img_path: str, device: torch.device, image_size=224) -> np.ndarray:
    img = Image.open(img_path).convert('L')
    x = SignatureDeterministicTransform(image_size)(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z = encoder(x).detach().cpu().numpy()[0]  # 已归一化
    return z


# ============ 原型库 ============

class PrototypeStore:
    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
    def _user_dir(self, uid: str) -> str:
        return os.path.join(self.registry_dir, uid)
    def save_user_proto(self, uid: str, name: str, proto: np.ndarray, num_samples: int):
        udir = self._user_dir(uid)
        os.makedirs(udir, exist_ok=True)
        np.save(os.path.join(udir, "prototype.npy"), proto.astype(np.float32))
        meta = {"uid": uid, "name": name, "num_samples": int(num_samples)}
        with open(os.path.join(udir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    def load_user_proto(self, uid: str):
        udir = self._user_dir(uid)
        proto_path = os.path.join(udir, "prototype.npy")
        meta_path = os.path.join(udir, "metadata.json")
        if not os.path.isfile(proto_path) or not os.path.isfile(meta_path):
            return None, None
        proto = np.load(proto_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return proto, meta
    def list_users(self) -> List[str]:
        return [d for d in os.listdir(self.registry_dir) if os.path.isdir(os.path.join(self.registry_dir, d))]
    def save_threshold(self, thr: float):
        with open(os.path.join(self.registry_dir, "threshold.txt"), "w") as f:
            f.write(f"{thr:.6f}\n")
    def load_threshold(self) -> Optional[float]:
        p = os.path.join(self.registry_dir, "threshold.txt")
        if not os.path.isfile(p):
            return None
        try:
            with open(p, "r") as f:
                return float(f.read().strip())
        except Exception:
            return None


# ============ 业务逻辑 ============

def register_user(model_path: str, registry_dir: str, uid: str, name: str, img_paths: List[str], image_size=224, device_str='cuda'):
    if len(img_paths) == 0:
        raise ValueError("No registration images provided.")
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder(model_path, device)
    # 计算嵌入，取均值并归一化
    feats = []
    for p in img_paths:
        if not os.path.isfile(p):
            print(f"[Warn] Image not found: {p}, skip.")
            continue
        f = compute_feature(encoder, p, device, image_size)
        feats.append(f)
    if len(feats) == 0:
        raise ValueError("No valid registration images after filtering.")
    mu = np.mean(np.stack(feats, axis=0), axis=0)
    mu = mu / (np.linalg.norm(mu) + 1e-12)

    store = PrototypeStore(registry_dir)
    # 如果已有原型，做增量融合（简单加权平均）
    prev_proto, prev_meta = store.load_user_proto(uid)
    if prev_proto is not None and prev_meta is not None:
        n_prev = int(prev_meta.get("num_samples", 1))
        n_new = len(feats)
        mu = (prev_proto * n_prev + mu * n_new) / max(1, (n_prev + n_new))
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        total_samples = n_prev + n_new
    else:
        total_samples = len(feats)

    store.save_user_proto(uid, name, mu, total_samples)
    print(f"[Register] uid={uid}, name={name}, samples={total_samples}, proto_saved={os.path.join(store._user_dir(uid),'prototype.npy')}")
    # 如果阈值不存在，给一个默认值
    thr_cur = store.load_threshold()
    if thr_cur is None:
        default_thr = 0.65
        store.save_threshold(default_thr)
        print(f"[Register] No global threshold found. Set default thr={default_thr:.6f}")

def verify_signature(model_path: str, registry_dir: str, uid: str, img_path: str, thr: Optional[float] = None, image_size=224, device_str='cuda') -> bool:
    store = PrototypeStore(registry_dir)
    proto, meta = store.load_user_proto(uid)
    if proto is None:
        raise ValueError(f"User uid={uid} not found in registry.")
    if thr is None:
        thr_file = store.load_threshold()
        if thr_file is None:
            raise ValueError("Global threshold not set. Use set-threshold or pass --thr.")
        thr = thr_file
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    encoder = load_encoder(model_path, device)
    q = compute_feature(encoder, img_path, device, image_size)
    # 余弦相似度
    score = float(np.dot(q, proto) / (np.linalg.norm(q) * np.linalg.norm(proto) + 1e-12))
    passed = (score >= thr)
    print(f"[Verify] uid={uid} score={score:.6f} thr={thr:.6f} => {'True' if passed else 'False'}")
    return passed


# ============ CLI ============

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_reg = sub.add_parser("register", help="Register a user with genuine signature images")
    ap_reg.add_argument('--model_path', type=str, required=True)
    ap_reg.add_argument('--registry_dir', type=str, required=True)
    ap_reg.add_argument('--uid', type=str, required=True)
    ap_reg.add_argument('--name', type=str, required=True)
    ap_reg.add_argument('--imgs', type=str, nargs='+', required=True)
    ap_reg.add_argument('--image_size', type=int, default=224)
    ap_reg.add_argument('--device', type=str, default='cuda')

    ap_ver = sub.add_parser("verify", help="Verify an unknown signature against a registered uid")
    ap_ver.add_argument('--model_path', type=str, required=True)
    ap_ver.add_argument('--registry_dir', type=str, required=True)
    ap_ver.add_argument('--uid', type=str, required=True)
    ap_ver.add_argument('--img', type=str, required=True)
    ap_ver.add_argument('--thr', type=float, default=None, help='Override global threshold if provided')
    ap_ver.add_argument('--image_size', type=int, default=224)
    ap_ver.add_argument('--device', type=str, default='cuda')

    ap_thr = sub.add_parser("set-threshold", help="Set or update global threshold for verification")
    ap_thr.add_argument('--registry_dir', type=str, required=True)
    ap_thr.add_argument('--thr', type=float, required=True)

    args = ap.parse_args()

    if args.cmd == "register":
        register_user(args.model_path, args.registry_dir, args.uid, args.name, args.imgs, image_size=args.image_size, device_str=args.device)
    elif args.cmd == "verify":
        res = verify_signature(args.model_path, args.registry_dir, args.uid, args.img, thr=args.thr, image_size=args.image_size, device_str=args.device)
        # 返回码：True->0，False->1
        import sys
        sys.exit(0 if res else 1)
    elif args.cmd == "set-threshold":
        store = PrototypeStore(args.registry_dir)
        store.save_threshold(args.thr)
        print(f"[Threshold] Global threshold set to {args.thr:.6f} in {args.registry_dir}/threshold.txt")


if __name__ == "__main__":
    main()
