#!/usr/bin/env python3
"""Merge <model>.shardKofN.jsonl files back into <model>.jsonl for score.py."""
import collections, json, os, re, sys

EVAL = "/mnt/data/cvhci_video_understanding/eval"


def main():
    d = f"{EVAL}/results"
    groups = collections.defaultdict(list)
    for f in os.listdir(d):
        m = re.match(r"^(.+)\.shard(\d+)of(\d+)\.jsonl$", f)
        if m:
            groups[m.group(1)].append(f)
    for model, files in groups.items():
        out = f"{d}/{model}.jsonl"
        seen = {}
        if os.path.exists(out):
            for l in open(out):
                try:
                    r = json.loads(l)
                    if not r.get("error"):
                        seen[r["review_id"]] = r
                except Exception:
                    pass
        for f in files:
            for l in open(f"{d}/{f}"):
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if not r.get("error"):
                    seen[r["review_id"]] = r
        with open(out, "w") as fh:
            for r in seen.values():
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{model}: merged {len(files)} shards -> {len(seen)} questions")


if __name__ == "__main__":
    main()
