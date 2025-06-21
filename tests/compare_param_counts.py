# compare_param_counts.py

# Cell 0: shared 모듈 임포트용 경로 설정
import sys
import os

# 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
from models.fno.model import FNO
from models.unet.model import UNet3D

def count_parameters(model, name="Model"):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔍 {name}")
    print(f"📦 Total parameters: {total:,}")
    print(f"🧠 Trainable parameters: {trainable:,}\n")
    return total, trainable

if __name__ == "__main__":
    # FNO 모델 정의
    fno_model = FNO(
        in_channels=1,
        out_channels=1,
        modes1=32,
        modes2=32,
        modes3=32,
        width=128,
        lifting_channels=128,
        add_grid=True,
        activation=nn.ReLU()
    )
    fno_model.eval()

    # U-Net 모델 정의
    unet_model = UNet3D()
    unet_model.eval()

    # 파라미터 수 비교 출력
    count_parameters(fno_model, name="Fourier Neural Operator (FNO)")
    count_parameters(unet_model, name="U-Net 3D")
