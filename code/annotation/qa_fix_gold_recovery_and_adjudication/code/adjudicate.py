#!/usr/bin/env python3
"""Decide, per question, what is actually wrong with it and what can be done about it.

Three views are combined, and none of them is allowed to confirm itself:
  gold    -- recovered from the old corpus (its provenance is recorded)
  human   -- an annotator who watched the footage, knowing no gold
  gemini  -- gemini-3.1-pro-preview watching the same footage, knowing no gold
plus a blind gate that answers with no footage at all.

The rule that matters: gold is only overturned when the two independent viewers agree
with each other against it. Where they disagree with each other the question goes to a
second, letter-shuffled Gemini vote rather than being decided by the louder source.
"""
import json, os, collections

from floodgate import CAPTION_MODEL

QA = "/mnt/data/cvhci_video_understanding/qa_fix"
UNTRUSTED = {"Junwei Zheng"}     # 43.6% option overlap vs 37.5% random: no signal

# Annotators whose clear calls are strong enough to settle a question on their own.
# Measured 2026-09-16 on the 124 questions where the gold, gemini-3.1-pro and
# gemini-3.8-flash all agree and 3.8-flash's verdict is "clear" -- a label the annotator
# played no part in forming. Exact-match when they said "clear", with the 95% Wilson lower
# bound, against an 8.9% weighted random baseline:
#   Yufan Chen 100% (75.7)  Di 90% (59.6)  Ruiping Liu/ Qian Yin 80% (54.8)
#   Chen Zhang 76.5% (52.7)  Haiwen Sun 73.3% (48.0)  Chengzhi Wu 66.7% (43.7)
#   zhihang chen 60% (35.7)  Xiaoye Wang 75% (30.1)
# Lei is left out: 45.5%, lower bound 21.3% -- above chance, not enough to decide alone.
# Note Di marks 54% of questions ambiguous yet is 90% right when committing: strictness is
# not inaccuracy, and the two must not be conflated.
TRUSTED_HUMAN = {"Yufan Chen", "Di", "Ruiping Liu/ Qian Yin", "Chen Zhang",
                 "Haiwen Sun", "Chengzhi Wu", "zhihang chen", "Xiaoye Wang"}


def load():
    idx = {}
    for l in open(f"{QA}/master_index.jsonl"):
        r = json.loads(l)
        idx[r["review_id"]] = r
    vid = collections.defaultdict(dict)
    for name in ("video_review.jsonl",):
        p = f"{QA}/{name}"
        if not os.path.exists(p):
            continue
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error"):
                vid[r["review_id"]][r["vote"]] = r
    # A vote from a DIFFERENT model generation. v0/v1 are the same model re-reading its own
    # shuffle, so their agreement is self-consistency; this one is independent evidence.
    # Calibrated 2026-09-16 on 160 questions: 93.0% on the three-way-agreed easy set, and
    # 41.7% (49.0% when its verdict is "clear") on the 60 questions where the human and the
    # gold agree but v0 got it wrong -- against a 9.0% weighted random baseline.
    cross = {}
    p = f"{QA}/video_review_x38.jsonl"
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error") and r.get("model") and r["model"] != CAPTION_MODEL:
                cross[r["review_id"]] = r
    blind = {}
    p = f"{QA}/blind_gate.jsonl"
    if os.path.exists(p):
        for l in open(p):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if not r.get("error"):
                blind[r["review_id"]] = r
    return idx, vid, blind, cross



def human_decides(out, h, n, gold):
    """A trusted annotator who watched the footage and was unambiguous settles the question.

    Used only where the automated rule would otherwise throw the question away. The person
    is the only viewer here who is not part of the family that wrote this corpus, so where
    a deadlock is between them and the models, they carry it.
    """
    if not h or h.get("annotator") not in TRUSTED_HUMAN or h.get("verdict") != "clear":
        return False
    ha = set(h.get("final_answer") or [])
    if len(ha) != n:
        return False
    out.setdefault("reasons", []).append("trusted_human_clear:" + h["annotator"])
    out["human_decided_by"] = h["annotator"]
    if gold and ha == gold:
        out["decision"] = "KEEP_GOLD_HUMAN_DECIDES"
    else:
        out["decision"] = "REKEY_HUMAN"
        out["new_gold"] = sorted(ha)
    return True



CROSS_MIN_VERDICT = "clear"      # the verdict filter is worth 5-7 points in both calib sets


def cross_break(out, cx, n, gold, ha, ga):
    """Third opinion, from a different model, used ONLY where the first two viewers
    deadlocked. It can promote an UNRESOLVED question; it can never overturn one that the
    existing two-viewer rule already settled, so nothing already delivered moves.

    Accepted only in the two shapes where the third vote is independent of what it is
    being compared against:
      * it lands on the human's answer -- two viewers of different kinds, neither of which
        saw the other's work, agreeing;
      * it lands on the gold key -- the same standard KEEP_GEMINI_ONLY already applies,
        but from a model that did not author this corpus.
    A third vote that merely agrees with v0/v1 is NOT accepted: those are the same model,
    and this project already measured that its self-agreement does not predict correctness
    (unanimous 39.4% vs split 40.5% on rbma273).
    """
    if not cx or cx.get("verdict") != CROSS_MIN_VERDICT:
        return False
    cxa = set(cx["final_answer"])
    if len(cxa) != n:
        return False
    out.setdefault("reasons", []).append("crossmodel:" + "".join(sorted(cxa)))
    out["crossmodel_answer"] = sorted(cxa)
    out["crossmodel_model"] = cx.get("model")
    if ha is not None and cxa == ha:
        if cxa == gold:
            out["decision"] = "KEEP_CONFIRMED_CROSSMODEL"
        else:
            out["decision"] = "REKEY_CROSSMODEL"; out["new_gold"] = sorted(cxa)
        return True
    if gold and cxa == gold:
        if cxa == ga:
            # v0 already said gold and a human who watched the footage said otherwise. A
            # third vote from the same family siding with gold is not a new viewer, it is
            # the same viewpoint louder -- and the gold here was itself Gemini-authored.
            # Record it, leave the question unresolved, let a human break the tie.
            out["reasons"].append("crossmodel_backs_gold_with_v0_human_dissents")
            out["decision"] = "UNRESOLVED_MODELS_FAVOR_GOLD"
            return False
        # v0 landed elsewhere, so this is an independent corroboration of the key
        out["decision"] = "KEEP_GOLD_CROSSMODEL"
        return True
    # agrees only with the same-model votes: recorded, not promoted
    if cxa == ga:
        out["reasons"].append("crossmodel_agrees_v0_only")
    return False


def decide(rec, votes, blind, cx=None):
    n = rec["n_select"]
    gold = set(rec["gold"])
    hs = [h for h in rec["human"] if h["annotator"] not in UNTRUSTED]
    h = hs[0] if hs else None
    v0 = votes.get(0)
    v1 = votes.get(1)
    out = {"review_id": rec["review_id"], "bundle": rec["bundle"], "source": rec["source"],
           "question_type": rec["question_type"], "n_select": n,
           "gold": sorted(gold), "gold_provenance": rec["gold_provenance"],
           "guessable": (blind or {}).get("guessable"),
           "human_verdict": h["verdict"] if h else None,
           "human_answer": sorted(h["final_answer"]) if h else None,
           "gemini_verdict": v0["verdict"] if v0 else None,
           "gemini_answer": v0["final_answer"] if v0 else None,
           "gemini_answer2": v1["final_answer"] if v1 else None}
    if not v0:
        out["decision"] = "PENDING_VIDEO"
        return out

    ga = set(v0["final_answer"])
    supported = {l for l, s in v0["per_option"].items() if s == "supported"}
    gold_ruled_by_gemini = {l for l in gold if v0["per_option"].get(l) == "ruled_out"}
    hr = {l for l, s in (h["per_option"] or {}).items() if s == "ruled_out"} if h else set()
    gold_ruled_by_human = gold & hr
    ha = set(h["final_answer"]) if h else None

    reasons = []
    if v0["verdict"] == "cant_tell":
        reasons.append("gemini_cant_tell")
    if h and h["verdict"] == "cant_tell":
        reasons.append("human_cant_tell")
    if v0["verdict"] == "ambiguous":
        reasons.append("gemini_ambiguous")
    if h and h["verdict"] == "ambiguous":
        reasons.append("human_ambiguous")
    if len(supported) < n:
        reasons.append(f"only_{len(supported)}_supported_of_{n}")
    if gold_ruled_by_gemini:
        reasons.append("gemini_rules_out_gold:" + "".join(sorted(gold_ruled_by_gemini)))
    if gold_ruled_by_human:
        reasons.append("human_rules_out_gold:" + "".join(sorted(gold_ruled_by_human)))
    if out["guessable"]:
        reasons.append("blind_guessable")
    out["reasons"] = reasons

    # no gold at all -> the video pass is the only candidate answer we have
    if not gold:
        if ha is not None and ha == ga and len(ga) == n and v0["verdict"] == "clear":
            out["decision"] = "KEY_FROM_AGREEMENT"; out["new_gold"] = sorted(ga)
        elif v1 and set(v1["final_answer"]) == ga and len(ga) == n:
            out["decision"] = "KEY_FROM_GEMINI2"; out["new_gold"] = sorted(ga)
        else:
            out["decision"] = "NEEDS_VOTE2" if not v1 else "NO_GOLD_UNRESOLVED"
            if out["decision"] == "NO_GOLD_UNRESOLVED":
                human_decides(out, h, n, gold) or cross_break(out, cx, n, gold, ha, ga)
        return out

    # the question has no valid answer at all: nothing in the option list survives
    if len(supported) == 0 and (h is None or len(set(h["final_answer"])) < n or h["verdict"] != "clear"):
        out["decision"] = "BROKEN_NO_CORRECT_OPTION"
        return out

    # both independent viewers land on the same answer
    if ha is not None and ha == ga and len(ga) == n:
        if ga == gold:
            out["decision"] = "KEEP_CONFIRMED"
        else:
            out["decision"] = "REKEY"; out["new_gold"] = sorted(ga)
        return out

    # only gemini has been here (no human coverage)
    if h is None:
        if ga == gold and v0["verdict"] == "clear" and not gold_ruled_by_gemini:
            out["decision"] = "KEEP_GEMINI_ONLY"
        elif v1 is not None:
            if set(v1["final_answer"]) == ga == gold:
                out["decision"] = "KEEP_GEMINI_ONLY"
            elif set(v1["final_answer"]) == ga and len(ga) == n and gold_ruled_by_gemini:
                out["decision"] = "REKEY_GEMINI_ONLY"; out["new_gold"] = sorted(ga)
            else:
                out["decision"] = "UNRESOLVED"
                cross_break(out, cx, n, gold, ha, ga)
        else:
            out["decision"] = "NEEDS_VOTE2"
        return out

    # human and gemini disagree with each other
    if ga == gold and h["verdict"] != "clear":
        out["decision"] = "KEEP_GOLD_HUMAN_UNSURE"
        return out
    if ha == gold:
        out["decision"] = "KEEP_GOLD_HUMAN_CONFIRMS"
        return out
    if gold_ruled_by_gemini and gold_ruled_by_human:
        out["decision"] = "GOLD_CONTRADICTED_NO_REPLACEMENT"
        # both viewers reject the key, but one of them also named a replacement
        human_decides(out, h, n, gold)
        return out
    out["decision"] = "NEEDS_VOTE2" if not v1 else "UNRESOLVED"
    if out["decision"] == "UNRESOLVED":
        human_decides(out, h, n, gold) or cross_break(out, cx, n, gold, ha, ga)
    return out


def rescore_blind(row, b):
    """The blind gate scored its votes against master_index's gold -- which is exactly the
    key a REKEY decision overturns, so a rekeyed question's guessable flag was measured
    against an answer we have since rejected. The votes themselves are still good: score
    them again against the key this adjudication just settled on."""
    if not b or not b.get("votes"):
        return
    key = set(row.get("new_gold") or row.get("gold") or [])
    if not key:
        row["guessable"] = None
        return
    hits = sum(1 for v in b["votes"] if set(v) == key)
    row["blind_hits"] = hits
    row["guessable"] = hits >= 2 if len(b["votes"]) >= 3 else hits >= 1
    reasons = row.get("reasons")
    if reasons is None:
        return
    if row["guessable"] and "blind_guessable" not in reasons:
        reasons.append("blind_guessable")
    elif not row["guessable"] and "blind_guessable" in reasons:
        reasons.remove("blind_guessable")


def main():
    idx, vid, blind, cross = load()
    rows = [decide(rec, vid.get(rid, {}), blind.get(rid), cross.get(rid))
            for rid, rec in idx.items()]
    for r in rows:
        rescore_blind(r, blind.get(r["review_id"]))
    with open(f"{QA}/adjudication.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["decision"] for r in rows)
    done = sum(v for k, v in c.items() if k != "PENDING_VIDEO")
    print(f"decided {done} / {len(rows)}")
    for k, v in c.most_common():
        print(f"  {k:34s} {v:5d}  {100*v/max(1,done):5.1f}%" if k != "PENDING_VIDEO" else f"  {k:34s} {v:5d}")
    rc = collections.Counter(x for r in rows if r.get("reasons") for x in r["reasons"])
    print("\nreasons (on decided):")
    for k, v in rc.most_common(14):
        print(f"  {k:44s} {v}")


if __name__ == "__main__":
    main()
