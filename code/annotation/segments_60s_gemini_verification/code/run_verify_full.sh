#!/bin/bash
# 全量核验: 对6452题池里剩余未核验的4901题全部跑Gemini视觉核验。
# 用户明确要求扩大到全部剩余题目, 而不是停在1551题的抽样估计上。
cd /mnt/task_runtime
D=/mnt/tmp/claude-0/-mnt-task-runtime/530b5738-50c5-4511-a092-b0779eaf18d9/scratchpad

echo "=== $(date) 开始切片 (4901题) ===" >> "$D/verify_full.log"
python3 cut_clips.py verify_candidates_remaining.jsonl "$D/clips_verify" >> "$D/cut_verify_full.log" 2>&1
echo "=== $(date) 切片完成: $(ls "$D/clips_verify" | wc -l) 个(累计) ===" >> "$D/verify_full.log"

echo "=== $(date) 开始核验 ===" >> "$D/verify_full.log"
RB_WORKERS=6 python3 visual_verify.py verify_candidates_remaining.jsonl "$D/clips_verify" verify_results.jsonl >> "$D/verify_full.log" 2>&1
echo "=== $(date) 全量核验完成 ===" >> "$D/verify_full.log"
