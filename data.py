"""FLAIR/AI-HUB 타일 로딩, 모델 로딩, 추론 + IoU."""
import json
import re
from pathlib import Path

import numpy as np

IGNORE = 255


# ---------------------------------------------------------------- files
def _read_tif(path):
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read()                      # (C, H, W)
    except ImportError:
        import tifffile
        a = tifffile.imread(path)
        if a.ndim == 2:
            return a[None]
        return a.transpose(2, 0, 1) if a.shape[-1] < a.shape[0] else a


def pair_files(img_dir, msk_dir, img_glob="IMG_*.tif", msk_glob="MSK_*.tif"):
    """IMG_000123.tif <-> MSK_000123.tif 처럼 파일명 마지막 숫자로 매칭."""
    def key(p):
        nums = re.findall(r"\d+", p.stem)
        return nums[-1] if nums else p.stem
    imgs = {key(p): p for p in sorted(Path(img_dir).rglob(img_glob))}
    msks = {key(p): p for p in sorted(Path(msk_dir).rglob(msk_glob))}
    common = sorted(set(imgs) & set(msks))
    return [(imgs[k], msks[k]) for k in common]


def build_lut(label_map_path, n_classes):
    """원본 마스크 값 -> 모델 클래스 인덱스. 매핑 없는 값은 IGNORE."""
    lut = np.full(256, IGNORE, np.uint8)
    if label_map_path:
        for k, v in json.load(open(label_map_path)).items():
            if not k.startswith("_"):
                lut[int(k)] = IGNORE if v is None else int(v)
    else:  # 기본값: 1..n -> 0..n-1  (반드시 모델 카드와 대조할 것)
        for v in range(1, n_classes + 1):
            lut[v] = v - 1
    return lut


def load_tiles(pairs, lut, bands=(0, 1, 2)):
    imgs, msks = [], []
    for ip, mp in pairs:
        img = _read_tif(ip)[list(bands)]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        imgs.append(img)
        msks.append(lut[_read_tif(mp)[0].astype(np.int64) % 256])
    return np.stack(imgs), np.stack(msks)


def channel_stats(imgs):
    """타일 전체 픽셀 기준 채널별 mean/std -> theta_base."""
    n = s = ss = 0.0
    for img in imgs:
        x = img.reshape(img.shape[0], -1).astype(np.float64)
        n += x.shape[1]
        s = s + x.sum(1)
        ss = ss + (x ** 2).sum(1)
    mean = s / n
    return np.concatenate([mean, np.sqrt(ss / n - mean ** 2)])


# ---------------------------------------------------------------- model
def load_model(ckpt, encoder="resnet34", n_classes=15, arch="Unet"):
    import segmentation_models_pytorch as smp
    import torch

    model = getattr(smp, arch)(encoder_name=encoder, encoder_weights=None,
                               in_channels=3, classes=n_classes)
    if str(ckpt).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(ckpt)
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        for k in ("state_dict", "model_state_dict", "model"):
            if isinstance(sd, dict) and isinstance(sd.get(k), dict):
                sd = sd[k]
                break

    target = model.state_dict()
    fixed = {}
    for k, v in sd.items():          # "model.", "module.", "seg_model." 같은 접두어 제거
        kk = k
        for _ in range(3):
            if kk in target:
                break
            kk = kk.split(".", 1)[1] if "." in kk else kk
        if kk in target and target[kk].shape == v.shape:
            fixed[kk] = v
    ratio = len(fixed) / len(target)
    print(f"[model] 매칭된 가중치 {len(fixed)}/{len(target)} ({ratio:.0%})")
    if ratio < 0.95:
        raise RuntimeError(
            "체크포인트 키가 모델 구조와 안 맞음. encoder/arch/n_classes 확인 필요.\n"
            f"  ckpt 예시 키: {list(sd)[:5]}\n  모델 예시 키: {list(target)[:5]}")
    model.load_state_dict(fixed, strict=False)
    return model.eval()


# ---------------------------------------------------------------- eval
class Evaluator:
    def __init__(self, model, imgs, n_classes, batch=8, device="cpu", threads=None):
        import torch
        if threads:
            torch.set_num_threads(threads)
        self.torch, self.device = torch, device
        self.model = model.to(device)
        self.imgs = torch.from_numpy(imgs)
        self.n_classes, self.batch = n_classes, batch

    def predict(self, theta):
        t = self.torch
        mean = t.tensor(theta[:3], dtype=t.float32).view(1, 3, 1, 1)
        std = t.tensor(theta[3:], dtype=t.float32).view(1, 3, 1, 1)
        N, _, H, W = self.imgs.shape
        out = np.empty((N, H, W), np.uint8)
        with t.inference_mode():
            for i in range(0, N, self.batch):
                x = ((self.imgs[i:i + self.batch].float() - mean) / std).to(self.device)
                out[i:i + self.batch] = self.model(x).argmax(1).cpu().numpy()
        return out

    def predict_per_tile(self):
        """타일마다 자기 채널 평균·표준편차로 정규화해서 추론.
        논문 표 5·6의 '(b) 영상별(image-wise) RGB 평균과 표준편차' 기준선의 한 가지 해석."""
        t = self.torch
        N, _, H, W = self.imgs.shape
        out = np.empty((N, H, W), np.uint8)
        with t.inference_mode():
            for i in range(0, N, self.batch):
                x = self.imgs[i:i + self.batch].float()
                m = x.mean(dim=(2, 3), keepdim=True)
                s = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
                out[i:i + self.batch] = self.model(((x - m) / s).to(self.device)).argmax(1).cpu().numpy()
        return out


def confusion(pred, gt, n):
    v = gt != IGNORE
    idx = gt[v].astype(np.int64) * n + pred[v].astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def class_iou(pred, gt, n):
    cm = confusion(pred, gt, n)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan) * 100


def decision_fusion(base_pred, opt_preds):
    """opt_preds: [(class_idx, pred), ...] 우선순위 낮은 것부터.
    뒤에 오는(=IoU 개선폭 큰) 클래스가 픽셀을 덮어씀."""
    out = base_pred.copy()
    for c, p in opt_preds:
        out[p == c] = c
    return out