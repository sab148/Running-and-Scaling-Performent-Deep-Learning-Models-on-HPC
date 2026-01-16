#!/usr/bin/env bash
source /p/scratch/atmlaml/HPC-Supporters-Course/sc_venv_template_HPC_supporter_course/venv/bin/activate

SCRIPT_NAME=$1
shift

nsys profile \
    --duration=30 \
    --delay=200 \
    --gpu-metrics-device=all \
    --nic-metrics=true \
    --stop-on-exit=false \
    --trace=nvtx,cuda,osrt \
    --python-sampling=true \
    --python-sampling-frequency=1 \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --python-functions-trace=profiler/config/profiling.json \
    --output=nsys_logs/nsys_logs_rank_${RANK} \
    --python-backtrace=cuda \
    --cudabacktrace=all \
    python -u "$SCRIPT_NAME" "$@"
