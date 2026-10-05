"""데이터·모델 입출력과 IoU 계산.

- 타일 읽기: FLAIR(IMG_/MSK_ 파일)와 AI-HUB(LC_..._연도.tif) 영상·라벨을 번호로 짝짓는다.
- 라벨 매핑: 데이터셋 원래 클래스 값을 모델 클래스 인덱스로 바꾼다(labelmap_*.json).
- 모델 로딩: FLAIR U-Net 가중치(ResNet34, MiT-B5)를 불러와 고정(eval)한다.
- 추론: θ(평균 3개, 표준편차 3개)로 입력을 정규화해 U-Net에 넣는다. 논문에서 바꾸는 값은 이 θ뿐이다.
- 평가: 타일 전체 픽셀을 합산한 혼동행렬로 클래스별 IoU를 계산한다.
- 결정 융합: 클래스별 최적 θ로 낸 예측을 기본 예측 위에 덮어쓴다(논문 12쪽).
"""
import json
import re
from pathlib import Path

import numpy as np

IGNORE = 255   # 평가에서 제외하는 라벨 값 (매핑 없는 클래스, AI-HUB 미분류 등)


# ---------------------------------------------------------------- files
def _read_tif(path):
    """GeoTIFF를 (채널, 높이, 너비) 배열로 읽는다. rasterio가 없으면 tifffile로 읽는다."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read()                      # (C, H, W)
    except ImportError:
        import tifffile
        a = tifffile.imread(path)
        if a.ndim == 2:
            return a[None]
        # tifffile은 (H, W, C)로 줄 때가 있어 채널 축을 앞으로 옮긴다
        return a.transpose(2, 0, 1) if a.shape[-1] < a.shape[0] else a


def tile_key(p):
    """파일 이름에서 타일 번호를 뽑는다. 영상·라벨 짝짓기와 --tiles 필터에 쓴다.

    FLAIR:  IMG_000123 -> 000123 (마지막 숫자).
    AI-HUB: LC_GG_AP12_0033_2017 -> 0033 (마지막 숫자는 촬영 연도라서 그 앞 숫자).
    """
    stem = Path(p).stem
    m = re.match(r"LC_[A-Z]+_(?:AP|SN)\d+_(\d+)_\d{4}$", stem)
    if m:
        return m.group(1)
    nums = re.findall(r"\d+", stem)
    return nums[-1] if nums else stem


def pair_files(img_dir, msk_dir, img_glob="IMG_*.tif", msk_glob="MSK_*.tif"):
    """영상 폴더와 라벨 폴더에서 같은 타일 번호끼리 짝지어 [(영상, 라벨), ...]을 돌려준다.

    FLAIR: IMG_000123.tif <-> MSK_000123.tif.
    AI-HUB는 영상·라벨 파일명이 같으므로 폴더를 나눠 두고 --img-glob/--msk-glob 을 LC_*.tif 로 준다.
    한쪽에만 있는 번호는 버린다.
    """
    key = tile_key
    imgs = {key(p): p for p in sorted(Path(img_dir).rglob(img_glob))}
    msks = {key(p): p for p in sorted(Path(msk_dir).rglob(msk_glob))}
    common = sorted(set(imgs) & set(msks))
    return [(imgs[k], msks[k]) for k in common]


def build_lut(label_map_path, n_classes):
    """원본 마스크 값(0~255) -> 모델 클래스 인덱스 변환표(길이 256)를 만든다.

    label_map_path(JSON)가 있으면 {"원본값": 모델 인덱스 또는 null}을 따른다.
    "_"로 시작하는 키는 설명용이라 건너뛴다. 매핑 없는 값은 IGNORE.
    없으면 FLAIR 기본 규칙(원본 1..n -> 0..n-1)을 쓴다.
    """
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
    """짝지은 파일을 모두 메모리에 올린다.

    bands: 영상에서 쓸 채널 번호. FLAIR 5밴드(R, G, B, NIR, 고도) 중 기본은 RGB(0, 1, 2).
    반환: 영상 (N, 3, H, W) uint8, 라벨 (N, H, W) 모델 인덱스(lut 적용 후).
    """
    imgs, msks = [], []
    for ip, mp in pairs:
        img = _read_tif(ip)[list(bands)]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        imgs.append(img)
        msks.append(lut[_read_tif(mp)[0].astype(np.int64) % 256])
    return np.stack(imgs), np.stack(msks)


def channel_stats(imgs):
    """타일 전체 픽셀을 합쳐 채널별 평균·표준편차를 계산한다 -> 기준 θ₀(θbase).

    타일마다 따로 구해 평균내는 것이 아니라 모든 픽셀을 한 집합으로 본다(pooled).
    E2·E5에서 이 방식이 논문 기준값과 맞았다.
    반환: [R 평균, G 평균, B 평균, R 표준편차, G 표준편차, B 표준편차]
    """
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
    """FLAIR 사전학습 U-Net을 불러와 가중치를 고정(eval)한 상태로 돌려준다.

    ckpt: 가중치 파일(.pth 또는 .safetensors). 예: FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth
    encoder: "resnet34" 또는 "mit_b5". 입력은 RGB 3채널(in_channels=3)로 고정.
    체크포인트 키 앞의 "model.", "module." 같은 접두어는 떼고 모델 구조에 맞춘다.
    95% 미만이 맞으면 구조가 다른 것이므로 멈춘다.
    """
    import segmentation_models_pytorch as smp
    import torch

    model = getattr(smp, arch)(encoder_name=encoder, encoder_weights=None,
                                in_channels=3, classes=n_classes)
    if str(ckpt).endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(ckpt)
    else:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        # 학습 프레임워크마다 가중치를 감싸는 키가 달라서 안쪽 dict를 꺼낸다
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
    return model.eval()     # 학습하지 않는다. 가중치는 끝까지 고정


# ---------------------------------------------------------------- eval
class Evaluator:
    """영상 묶음을 한 번 올려두고 θ만 바꿔 가며 반복 추론하는 도구.

    탐색 중 추론 1회 = predict(θ) 1회다. 영상은 uint8 그대로 두고 배치마다 정규화한다.
    """

    def __init__(self, model, imgs, n_classes, batch=8, device="cpu", threads=None):
        import torch
        if threads:
            torch.set_num_threads(threads)
        self.torch, self.device = torch, device
        self.model = model.to(device)
        self.imgs = torch.from_numpy(imgs)
        self.n_classes, self.batch = n_classes, batch

    def predict(self, theta):
        """θ 하나로 모든 타일을 정규화해 추론한다.

        정규화: x' = (x − θ평균) / θ표준편차 (채널별). 논문에서 최적화하는 것은 이 6개 값이다.
        반환: (N, H, W) 픽셀별 예측 클래스 인덱스.
        """
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
        """타일마다 자기 채널 평균·표준편차로 정규화해서 추론한다.

        논문 표 5·6의 '(b) 영상별(image-wise) RGB 평균과 표준편차' 기준선의 한 가지 해석.
        (E2에서는 타일 전체 통계 한 세트(channel_stats)가 논문 값과 맞았다.)
        """
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
    """혼동행렬 (n×n). 행 = 정답, 열 = 예측. IGNORE 픽셀은 뺀다. 전체 타일 픽셀을 합산한다."""
    v = gt != IGNORE
    idx = gt[v].astype(np.int64) * n + pred[v].astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def class_iou(pred, gt, n):
    """클래스별 IoU(%) = TP / (TP + FP + FN) × 100 (논문 3.2절).

    타일별 IoU를 평균내지 않고, 전체 픽셀 혼동행렬에서 한 번에 계산한다
    (저자 E2 지도로 확인한 방식과 같다). 정답·예측 모두 없는 클래스는 NaN.
    """
    cm = confusion(pred, gt, n)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1) - tp          # TP + FP + FN
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan) * 100


def decision_fusion(base_pred, opt_preds):
    """결정 융합 (논문 4.1.1절, 12쪽).

    base_pred: 기준 θ로 낸 예측.
    opt_preds: [(클래스 인덱스, 그 클래스의 최적 θ로 낸 예측), ...] 우선순위 낮은 것부터.
    각 최적 예측에서 그 클래스로 칠해진 픽셀만 기본 예측 위에 덮어쓴다.
    뒤에 오는(= IoU 개선폭이 큰) 클래스가 나중에 덮어써서 우선한다.
    덮어쓰기만 하므로 기본 예측의 오탐(그 클래스로 잘못 칠한 픽셀)은 지우지 못한다.
    """
    out = base_pred.copy()
    for c, p in opt_preds:
        out[p == c] = c
    return out
