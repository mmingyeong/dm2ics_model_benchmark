
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

