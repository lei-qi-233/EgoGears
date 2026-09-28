#!/bin/bash
# 第一轮视觉核验: 切齐候选题的片段, 然后逐选项对账。
cd /mnt/task_runtime
D=/mnt/tmp/claude-0/-mnt-task-runtime/530b5738-50c5-4511-a092-b0779eaf18d9/scratchpad

python3 cut_clips.py verify_candidates.jsonl "$D/clips_verify" >> "$D/cut_verify.log" 2>&1
echo "切片完成: $(ls "$D/clips_verify" | wc -l) 个"

RB_WORKERS=6 python3 visual_verify.py verify_candidates.jsonl "$D/clips_verify" verify_results.jsonl
echo "核验完成"
