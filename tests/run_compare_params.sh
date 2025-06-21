#!/bin/bash
#PBS -N param_compare
#PBS -q long
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l mem=8gb
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -V
#PBS -m abe
#PBS -M mmingyeong@kasi.re.kr

cd $PBS_O_WORKDIR

# 로그 디렉토리 생성
export LOGDIR=/home/users/mmingyeong/_dm2ics_model_benchmark/dm2ics_model_benchmark/tests/logs
mkdir -p "$LOGDIR"

# 로그 파일 경로 설정
LOGFILE="$LOGDIR/param_compare_${PBS_JOBID}.log"
exec > "$LOGFILE" 2>&1

echo "📌 Job ID: $PBS_JOBID"
echo "📁 Log File: $LOGFILE"

source ~/.bashrc
conda activate py312

echo "🚀 Starting parameter count comparison script on $(hostname) at $(date)"

python compare_param_counts.py

echo "✅ Script completed at $(date)"
