#!/bin/bash
#PBS -N fno_summary
#PBS -q long
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l mem=8gb
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -V

cd $PBS_O_WORKDIR

# 로그 디렉토리 생성
export LOGDIR=/home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/tests/logs
mkdir -p "$LOGDIR"

LOGFILE="$LOGDIR/fno_summary_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"

source ~/.bashrc
conda activate py312

echo "🚀 Running FNO parameter summary script on $(hostname) at $(date)"

python fno_summary.py

echo "✅ Script finished at $(date)"
