#!/bin/bash
cd /mnt/task_runtime
D=/mnt/tmp/claude-0/-mnt-task-runtime/530b5738-50c5-4511-a092-b0779eaf18d9/scratchpad
# 等切片跑完 (653 个片段)
while [ "$(ls $D/clips677 2>/dev/null | wc -l)" -lt 653 ]; do sleep 20; done
sleep 10
echo "切片完成: $(ls $D/clips677|wc -l) 个, 启动标注"
RB_CONC=4 python3 gemini_label_video.py gemini-3.1-pro-preview segments_60s_subset.jsonl $D/clips677 gemini_labels/vid_pro_677.jsonl > gemini_labels/vid_pro.log 2>&1 &
P1=$!
sleep 5
RB_CONC=4 python3 gemini_label_video.py gemini-3.8-flash segments_60s_subset.jsonl $D/clips677 gemini_labels/vid_flash_677.jsonl > gemini_labels/vid_flash.log 2>&1 &
P2=$!
wait $P1 $P2
echo "两个视频标注全部完成"
