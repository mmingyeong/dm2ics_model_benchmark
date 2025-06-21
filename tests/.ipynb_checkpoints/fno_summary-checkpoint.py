# fno_summary.py

# Cell 0: 모듈 import를 위한 경로 설정
import os, sys
sys.path.append(os.path.abspath(".."))  # shared, models 디렉토리 접근 가능하도록 경로 추가

from models.fno.model import FNO
import torch
from torchinfo import summary
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FNO(
    in_channels=1,
    out_channels=1,
    modes1=32,
    modes2=32,
    modes3=32,
    width=128,
    lifting_channels=128,
    add_grid=True,
    activation=nn.ReLU
).to(device)

model.train()
print("✅ FNO model loaded and set to training mode.")

summary(
    model,
    input_size=(2, 1, 60, 60, 60),
    col_names=["input_size", "output_size", "num_params", "kernel_size"],
    verbose=1
)
