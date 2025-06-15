#!/bin/bash
#PBS -N predict_fno
#PBS -q long
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -V

cd $PBS_O_WORKDIR

# 로그 디렉토리 생성
LOGDIR=/home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/scripts/logs
mkdir -p "$LOGDIR"

# 로그 파일명에 PBS 잡 ID 포함
LOGFILE="$LOGDIR/predict_fno_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"

source ~/.bashrc
conda activate py312

echo "🚀 Starting FNO prediction job on $(hostname) at $(date)"

python /home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/models/fno/predict.py \
  --input_dir /caefs/data/IllustrisTNG/subcube/input \
  --output_dir /caefs/data/IllustrisTNG/predictions/fno/Sample20_epoch50 \
  --model_path /home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/results/fno/fno_sample20_epoch50_best.pt \
  --batch_size 4

echo "✅ FNO prediction completed at $(date)"
