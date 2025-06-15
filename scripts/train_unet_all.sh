#!/bin/bash
#PBS -N train_unet
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -V

cd "$PBS_O_WORKDIR"

export LOGDIR=/home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/scripts/logs
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/train_unet_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"
echo "📂 Working Dir: $PBS_O_WORKDIR"
echo "🐍 Python Path: $(which python)"
echo "🧪 Python Version: $(python --version)"
nvidia-smi || echo "⚠️ No GPU detected or nvidia-smi not available"

source ~/.bashrc
conda activate py312

echo "🚀 Starting U-Net training job on $(hostname) at $(date)"

python /home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/models/unet/train.py \
  --input_path /caefs/data/IllustrisTNG/subcube/input \
  --output_path /caefs/data/IllustrisTNG/subcube/output \
  --batch_size 4 \
  --epochs 100 \
  --min_lr 1e-4 \
  --max_lr 1e-3 \
  --patience 10 \
  --alpha 0.05 \
  --ckpt_dir /home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/results/unet \
  --device cuda \
  --sample_fraction 1

echo "✅ Training completed at $(date)"
