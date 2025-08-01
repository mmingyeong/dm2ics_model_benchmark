#!/bin/bash
#PBS -N predict_vit3d
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M mmingyeong@kasi.re.kr

cd "$PBS_O_WORKDIR"

# 로그 디렉토리 생성
LOGDIR=/home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/scripts/logs
mkdir -p "$LOGDIR"

# 로그 파일명에 PBS 잡 ID 포함
LOGFILE="$LOGDIR/predict_vit3d_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"

source ~/.bashrc
conda activate py312

echo "🚀 Starting ViT3D prediction job on $(hostname) at $(date)"

python /home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/models/vit/predict.py \
  --input_dir /caefs/data/IllustrisTNG/subcube/input \
  --output_dir /caefs/data/IllustrisTNG/predictions/vit/Sample1_epoch30 \
  --model_path /home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/results/vit_test/vit_full_3dreg_sample1_epoch30_final.pt \
  --batch_size 2 \
  --image_size 60 \
  --frames 60 \
  --image_patch_size 4 \
  --frame_patch_size 4 \
  --emb_dim 128 \
  --depth 6 \
  --heads 8 \
  --mlp_dim 256

echo "✅ Prediction completed at $(date)"
