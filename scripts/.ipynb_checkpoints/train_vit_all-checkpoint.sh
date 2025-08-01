#!/bin/bash
#PBS -N test_vit3d
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M mmingyeong@kasi.re.kr

cd "$PBS_O_WORKDIR"

export LOGDIR=/home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/scripts/logs
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/test_vit3d_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"
echo "📂 Working Dir: $PBS_O_WORKDIR"
echo "🐍 Python Path: $(which python)"
echo "🧪 Python Version: $(python --version)"
nvidia-smi || echo "⚠️ No GPU detected or nvidia-smi not available"

source ~/.bashrc
conda activate py312

echo "🧪 Starting lightweight ViT3D test training at $(date)"

python /home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/models/vit/train.py \
  --input_path /caefs/data/IllustrisTNG/subcube/input \
  --output_path /caefs/data/IllustrisTNG/subcube/output \
  --batch_size 4 \
  --epochs 30 \
  --min_lr 1e-4 \
  --max_lr 1e-3 \
  --patience 10 \
  --ckpt_dir /home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/results/vit_test \
  --device cuda \
  --sample_fraction 0.01 \
  --model_name test \
  --image_size 128 \
  --frames 64 \
  --image_patch_size 16 \
  --frame_patch_size 8 \
  --emb_dim 256 \
  --depth 6 \
  --heads 8 \
  --mlp_dim 512

echo "✅ Test training completed at $(date)"
