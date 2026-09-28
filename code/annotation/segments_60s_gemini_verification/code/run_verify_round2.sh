#!/bin/bash
# 第二轮补充核验: 大类配额不够, 从剩余池补抽后核验。
cd /mnt/task_runtime
D=/mnt/tmp/claude-0/-mnt-task-runtime/530b5738-50c5-4511-a092-b0779eaf18d9/scratchpad

python3 cut_clips.py verify_candidates_round2.jsonl "$D/clips_verify" >> "$D/cut_verify_round2.log" 2>&1
echo "round2 切片完成: $(ls "$D/clips_verify" | wc -l) 个(累计)"

RB_WORKERS=6 python3 visual_verify.py verify_candidates_round2.jsonl "$D/clips_verify" verify_results.jsonl
echo "round2 核验完成"
