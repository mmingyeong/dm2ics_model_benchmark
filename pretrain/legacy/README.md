
# Retrain & Debug Pipeline  
# 재학습 및 디버그 파이프라인

## Models / 대상 모델
- GAN (3D Pix2PixCC-style cGAN)  
- UNet (3D V-Net style)  
- FNO  
- ViT3D  

## Purpose / 목적
Quickly sanity-check and tune each model before full training:  
- verify data, shapes, gradients  
- profile batch size / memory / LR schedule  
- decompose losses (e.g., GAN: LSGAN, FM, CCC)  
- visualize predictions vs targets  
그 다음에 안정된 설정으로 본격 재학습 수행.  
데이터/모델/학습 설정을 사전 점검하고 튜닝한 후 전체 학습을 진행.


## Workflow / 흐름

1. **Run debug notebook** (e.g., `debug_pipeline_gan.ipynb`)

   * overfit small batch, loss breakdown, grad norm, lr schedule, batch-size profiling, slice visualization.
2. **Tune key hyperparameters**

   * learning rate, loss weights (`lambda_*`), G/D balance (GAN), scheduler settings.
3. **Save baseline diagnostics**

   * CSV / TensorBoard logging.
4. **Launch full training** with tuned config.
5. **Monitor validation & checkpoints**

   * save best / final.
6. **Post-eval**: compare predictions (slices, power spectrum, metrics).


## Tips

* Overfit one batch first. / 먼저 한 배치에 overfit 해보라.
* Record baseline before changing anything. / 바꾸기 전에 기준 저장.
* Inspect loss components separately. / 손실 항을 분리해서 보라.
* Balance G/D if using GAN. / GAN이면 G/D 균형 신경 써라.

## Contact / 문의

Mingyeong Yang (양민경) — [mmingyeong@kasi.re.kr](mailto:mmingyeong@kasi.re.kr)


-----------------------------------

# Pretrain Debug Pipelines — What Each Pipeline Checks  
# 사전 훈련 디버그 파이프라인 — 각 파이프라인이 점검하는 항목

## Purpose / 목적
Sanity-check and surface failure modes before full-scale retraining.  
전체 재학습 전에 모델·데이터·학습 설정의 이상 징후를 조기 탐지하고 안정적인 시작점을 확보하기 위함.

---

## Common Checks Across All Models / 모든 모델 공통 점검

1. **Data sanity (shape / dtype / scale)**  
   - Inputs and targets have expected dimensions, channels, and numeric range.  
   - 입력/타깃의 shape, 채널, 자료형, 스케일이 기대에 맞는지 확인.

2. **Overfit small batch (“smoke test”)**  
   - Train on a tiny subset (one or few batches) to verify loss decreases and gradients flow.  
   - 아주 작은 배치에 overfit 시켜 손실이 줄어들고 역전파가 작동하는지 확인.

3. **Batch size vs memory/time profiling**  
   - Sweep batch sizes to find feasible trade-off between GPU memory and speed.  
   - 배치 크기에 따라 메모리 사용량 및 처리 시간 실험.

4. **Learning rate / scheduler behavior**  
   - Verify learning rate schedule (cosine, CLR, etc.) is doing what’s expected and correlate with loss changes.  
   - 학습률 스케줄이 의도대로 변하는지, 손실에 어떻게 반응하는지 점검.

5. **Gradient norms & stability**  
   - Track norms to detect vanishing/exploding gradients; optionally apply clipping.  
   - 그래디언트 크기를 모니터링해 소멸/폭발 이상 여부 확인 및 필요시 클리핑.

6. **Prediction vs target visualization**  
   - Slice-wise comparison, residuals, log transforms for inspecting output quality.  
   - 중간 슬라이스, 잔차 맵, 로그 변환을 통한 예측 품질 시각적 점검.

7. **Structural diagnostics (e.g., power spectrum)**  
   - Compare frequency-domain statistics to ensure physical/structural consistency.  
   - 예측과 타깃의 파워 스펙트럼 등을 비교해 구조적 차이 분석.

8. **Baseline logging**  
   - Record losses, lrs, gradient norms, example outputs to CSV / TensorBoard for later comparison.  
   - 기준선 데이터를 저장해 하이퍼파라미터 변경 시 효과를 평가할 수 있게 함.

---

## GAN Pipeline (3D Pix2PixCC-style cGAN) / GAN 파이프라인

- **Loss decomposition**: LSGAN term, Feature Matching term, Correlation/CCC term separately.  
  손실을 LSGAN, Feature Matching, CCC/Correlation으로 분해해 각각의 기여도 확인.  
- **G/D balance**: Monitor relative strength; optionally adjust update frequency, learning rates, or loss weights.  
  Generator vs Discriminator의 균형 감시 및 필요시 업데이트 비율이나 가중치 조정.  
- **Channel balancing logic validation**: If used, confirm real/fake pair construction behaves as expected.  
  채널 균형 로직이 제대로 real/fake 페어를 구성하는지 검증.  
- **Patch discriminator multi-scale outputs**: Inspect intermediate feature maps for real vs fake.  
  PatchGAN의 다중 스케일 출력(특징 맵)을 실제/생성 기준으로 비교.

---

## UNet Pipeline (3D V-Net style) / UNet 파이프라인

- **Skip connection consistency**: Ensure encoder-decoder concatenation shapes align and no mismatches occur.  
  인코더-디코더 스킵 연결 시 shape 일치 확인.  
- **Output identity behavior**: For regression, final activation is identity; check dynamic range and bias.  
  회귀 목적에 맞게 최종 출력이 비선형 없이 적절한 스케일을 갖는지 점검.  
- **BatchNorm/InstanceNorm statistics**: Verify normalization layers behave (running stats or absence if disabled).  
  정규화 계층이 정상적으로 작동하는지 (running stats, affine 등) 확인.

---

## FNO Pipeline / FNO 파이프라인

- **Fourier mode truncation impact**: Vary number of modes and inspect reconstruction fidelity.  
  Fourier mode 수를 바꿨을 때 복원 품질이 어떻게 달라지는지 비교.  
- **Spectral aliasing detection**: Ensure padding / resolution handling does not introduce artifacts in frequency domain.  
  주파수 도메인에서 aliasing이나 패딩 실수로 인한 왜곡이 없는지 점검.  
- **Operator stability**: Check sensitivity to input perturbation (small noise) to assess smoothness.  
  입력에 작은 잡음을 섞었을 때 출력이 지나치게 불안정하지 않은지 확인.

---

## ViT3D Pipeline / ViT3D 파이프라인

- **Patch tiling and reconstruction**: Confirm 3D patch splitting and positional embedding alignment.  
  3D 패치를 나누고 재조합할 때 위치 임베딩이 일관된지 점검.  
- **Sequence length vs capacity**: Verify model handles the expected token length without degradation.  
  시퀀스 길이(패치 개수)에 대해 모델이 처리할 수 있는지 확인.  
- **Attention maps inspection**: (Optional) Visualize attention to see if the model focuses on meaningful regions.  
  주의(attention) 맵을 시각화해 중요한 영역을 보는지 평가.

---

## Usage Note / 사용 참고

Each model has its own notebook implementing these checks; keep the logs separate per model for comparison.  
모든 점검은 각 모델별 노트북에서 실행하며, 결과 로그는 모델별로 분리해서 기록해두는 것이 나중에 비교에 유리하다.

---

## Contact / 문의

Mingyeong Yang (양민경) — mmingyeong@kasi.re.kr  
