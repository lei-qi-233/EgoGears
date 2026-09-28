#!/usr/bin/env python3
"""Assemble the repaired corpus.

Each question comes out with one of four statuses and the evidence behind it, so a
reader can disagree with any single call without having to re-run the whole audit.
Nothing is kept on the strength of a single source: a question is `confirmed` only when
two viewers that could not see each other's work land on the same answer, or when the
one viewer that exists is clear and nothing contradicts it.
"""
import json, os, collections

QA = "/mnt/data/cvhci_video_understanding/qa_fix"

KEEP = {"KEEP_CONFIRMED", "KEEP_GOLD_HUMAN_CONFIRMS", "KEEP_GOLD_HUMAN_UNSURE", "KEEP_GEMINI_ONLY",
        # settled by a third vote from a different model where the first two deadlocked
        "KEEP_CONFIRMED_CROSSMODEL", "KEEP_GOLD_CROSSMODEL",
        # settled by a trusted annotator who watched the footage
        "KEEP_GOLD_HUMAN_DECIDES"}
REKEY = {"REKEY", "KEY_FROM_AGREEMENT", "KEY_FROM_GEMINI2", "REKEY_GEMINI_ONLY",
         "REKEY_CROSSMODEL", "REKEY_HUMAN"}
DROPPABLE = {"BROKEN_NO_CORRECT_OPTION", "GOLD_CONTRADICTED_NO_REPLACEMENT",
             "UNRESOLVED", "NO_GOLD_UNRESOLVED", "NEEDS_VOTE2", "PENDING_VIDEO",
             "UNRESOLVED_MODELS_FAVOR_GOLD"}


def main():
    idx = {json.loads(l)["review_id"]: json.loads(l) for l in open(f"{QA}/master_index.jsonl")}
    adj = {json.loads(l)["review_id"]: json.loads(l) for l in open(f"{QA}/adjudication.jsonl")}
    rep = {}
    p = f"{QA}/repairs.jsonl"
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("repaired"):
                rep[r["review_id"]] = r
    vid = collections.defaultdict(dict)
    for l in open(f"{QA}/video_review.jsonl"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if not r.get("error"):
            vid[r["review_id"]][r["vote"]] = r

    out, stats, dropreason = [], collections.Counter(), collections.Counter()
    for rid, rec in idx.items():
        a = adj.get(rid, {"decision": "PENDING_VIDEO"})
        d = a["decision"]
        row = {"review_id": rid, "bundle": rec["bundle"], "source": rec["source"],
               "settled_by_crossmodel": d.endswith("_CROSSMODEL") or None,
               "crossmodel_answer": a.get("crossmodel_answer"),
               "crossmodel_model": a.get("crossmodel_model"),

               "unit": rec["unit"], "question_type": rec["question_type"],
               "n_select": rec["n_select"], "question": rec["question"],
               "options": rec["options"], "clips": rec["clips"],
               "decision": d, "reasons": a.get("reasons", []),
               "blind_guessable": a.get("guessable")}
        if rid in rep:
            r = rep[rid]
            row.update(status="repaired", answer=r["answer"], options=r["options"],
                       repair_rounds=r.get("rounds_used"),
                       evidence=r.get("evidence"), distractor_basis=r.get("distractor_basis"),
                       original_options=rec["options"], original_answer=rec["gold"])
        elif d in REKEY and not a.get("guessable"):
            # For a human-decided rekey the gemini evidence argues for the answer this
            # decision just rejected, so attaching it would document the wrong case. Use
            # what the person who set the key actually wrote.
            if a.get("human_decided_by"):
                hh = next((x for x in (rec.get("human") or [])
                           if x.get("annotator") == a["human_decided_by"]), {})
                ev = {"per_option": hh.get("per_option"), "notes": hh.get("notes"),
                      "annotator": a["human_decided_by"]}
            else:
                ev = (vid[rid].get(0) or {}).get("evidence")
            row.update(status="rekeyed", answer=a["new_gold"], original_answer=rec["gold"],
                       evidence=ev, evidence_source=("human" if a.get("human_decided_by") else "gemini_v0"))
            # only where the delivered key really is the annotator's: a repair, if one
            # survived, replaced the options this person was judging and takes priority
            # above, so carrying the name there would credit a call that was not used
            if a.get("human_decided_by"):
                row["human_decided_by"] = a["human_decided_by"]
        elif d in REKEY:
            # a rekeyed question is guessable against its NEW key and no repair survived:
            # the same rule the KEEP branch applies, applied here too
            row.update(status="dropped", answer=a["new_gold"], original_answer=rec["gold"],
                       drop_reason="blind_guessable_unrepaired")
        elif d in KEEP and not a.get("guessable"):
            row.update(status="confirmed", answer=rec["gold"],
                       evidence=(vid[rid].get(0) or {}).get("evidence"))
        elif d in KEEP and a.get("guessable"):
            row.update(status="dropped", answer=rec["gold"], drop_reason="blind_guessable_unrepaired")
        else:
            why = ("models_favor_gold_human_dissents" if d == "UNRESOLVED_MODELS_FAVOR_GOLD" else
                   "no_correct_option" if d == "BROKEN_NO_CORRECT_OPTION" else
                   "gold_contradicted_no_replacement" if d == "GOLD_CONTRADICTED_NO_REPLACEMENT" else
                   "footage_does_not_cover" if "gemini_cant_tell" in a.get("reasons", []) else
                   "not_reviewed" if d == "PENDING_VIDEO" else "unresolved_disagreement")
            row.update(status="dropped", answer=rec["gold"], drop_reason=why)
        stats[row["status"]] += 1
        if row["status"] == "dropped":
            dropreason[row["drop_reason"]] += 1
        out.append(row)

    with open(f"{QA}/runningbench_v2.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = [r for r in out if r["status"] != "dropped"]
    with open(f"{QA}/runningbench_v2_kept.jsonl", "w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"total {len(out)}")
    for k, v in stats.most_common():
        print(f"  {k:10s} {v:5d}  {100*v/len(out):5.1f}%")
    print("\ndropped because:")
    for k, v in dropreason.most_common():
        print(f"  {k:36s} {v:5d}")
    print(f"\nkept corpus: {len(kept)}")
    bt = collections.Counter(r["question_type"] for r in kept)
    for k, v in bt.most_common():
        print(f"  {k:34s} {v}")


if __name__ == "__main__":
    main()
