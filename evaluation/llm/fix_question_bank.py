#!/usr/bin/env python3
"""
Fix 26 questions in question_bank.json per the detailed specifications.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

INPUT = str(Path(__file__).resolve().parent / "question_bank.json")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

def find_q(questions, qid):
    for q in questions:
        if q["id"] == qid:
            return q
    raise KeyError(f"Question {qid} not found")

def replace_in_list(lst, old, new):
    """Replace old substring with new in every element of a list. Returns count of changes."""
    count = 0
    for i, item in enumerate(lst):
        if old in item:
            lst[i] = item.replace(old, new)
            count += 1
    return count

def main():
    data = load_json(INPUT)
    questions = data["questions"]
    changes_log = {}

    # =========================================================================
    # Q01
    # =========================================================================
    q = find_q(questions, "Q01")
    ch = []
    # expected_reasoning[0]: "A1 uses 20 CBC features" -> "A1 uses 39 features (20 CBC + 19 CMP)"
    old_val = q["expected_reasoning"][0]
    q["expected_reasoning"][0] = q["expected_reasoning"][0].replace(
        "A1 uses 20 CBC features", "A1 uses 39 features (20 CBC + 19 CMP)"
    )
    ch.append(f"expected_reasoning[0]: '{old_val}' -> '{q['expected_reasoning'][0]}'")

    # rubric.factual_accuracy: "20 features from CBC" -> "39 features (20 CBC + 19 CMP)"
    old_val = q["rubric"]["factual_accuracy"]
    q["rubric"]["factual_accuracy"] = q["rubric"]["factual_accuracy"].replace(
        "20 features from CBC", "39 features (20 CBC + 19 CMP)"
    )
    ch.append(f"rubric.factual_accuracy: '{old_val}' -> '{q['rubric']['factual_accuracy']}'")
    changes_log["Q01"] = ch

    # =========================================================================
    # Q02
    # =========================================================================
    q = find_q(questions, "Q02")
    ch = []
    old_val = q["rubric"]["factual_accuracy"]
    q["rubric"]["factual_accuracy"] = q["rubric"]["factual_accuracy"].replace("N=21", "N=28")
    ch.append(f"rubric.factual_accuracy: '{old_val}' -> '{q['rubric']['factual_accuracy']}'")
    changes_log["Q02"] = ch

    # =========================================================================
    # Q03
    # =========================================================================
    q = find_q(questions, "Q03")
    ch = []
    # ground_truth_key_facts[0]: "A1 (CBC, 20 features)" -> "A1 (CBC + CMP, 39 features)"
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "A1 (CBC, 20 features)" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace(
                "A1 (CBC, 20 features)", "A1 (CBC + CMP, 39 features)"
            )
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")

    # expected_reasoning: "20 CBC features" -> "39 features (20 CBC + 19 CMP)"
    for i, item in enumerate(q["expected_reasoning"]):
        if "20 CBC features" in item:
            old_val = item
            q["expected_reasoning"][i] = item.replace(
                "20 CBC features", "39 features (20 CBC + 19 CMP)"
            )
            ch.append(f"expected_reasoning[{i}]: '{old_val}' -> '{q['expected_reasoning'][i]}'")

    # expected_reasoning: "N=21" -> "N=28 (with ~21 training samples per LOCO fold)"
    for i, item in enumerate(q["expected_reasoning"]):
        if "N=21" in item:
            old_val = item
            q["expected_reasoning"][i] = item.replace(
                "N=21", "N=28 (with ~21 training samples per LOCO fold)"
            )
            ch.append(f"expected_reasoning[{i}]: '{old_val}' -> '{q['expected_reasoning'][i]}'")
    changes_log["Q03"] = ch

    # =========================================================================
    # Q04
    # =========================================================================
    q = find_q(questions, "Q04")
    ch = []
    for i, item in enumerate(q["expected_reasoning"]):
        if "21 samples" in item:
            old_val = item
            q["expected_reasoning"][i] = item.replace(
                "21 samples", "28 samples (~21 training per LOCO fold)"
            )
            ch.append(f"expected_reasoning[{i}]: '{old_val}' -> '{q['expected_reasoning'][i]}'")
    changes_log["Q04"] = ch

    # =========================================================================
    # Q06
    # =========================================================================
    q = find_q(questions, "Q06")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "N=21 total" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace("N=21 total", "N=28 total")
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q06"] = ch

    # =========================================================================
    # Q07
    # =========================================================================
    q = find_q(questions, "Q07")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "~21 samples" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace("~21 samples", "28 samples")
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q07"] = ch

    # =========================================================================
    # Q10
    # =========================================================================
    q = find_q(questions, "Q10")
    ch = []
    # question text: "RF performance (0.885" -> "RF performance (0.884"
    old_val = q["question"]
    q["question"] = q["question"].replace("0.885", "0.884")
    if q["question"] != old_val:
        ch.append(f"question: replaced 0.885 -> 0.884")

    # ground_truth_key_facts: fix B1 ablation scores
    for i, item in enumerate(q["ground_truth_key_facts"]):
        old_item = item
        changed = False
        # RF=0.885 -> RF=0.884
        if "RF=0.885" in item:
            item = item.replace("RF=0.885", "RF=0.884")
            changed = True
        # B1 All: MLP=0.839 -> MLP=0.854
        if "B1 All" in item and "MLP=0.839" in item:
            item = item.replace("MLP=0.839", "MLP=0.854")
            changed = True
        # B1 Effect-only: MLP=0.756 -> MLP=0.741
        if "Effect-only" in item and "MLP=0.756" in item:
            item = item.replace("MLP=0.756", "MLP=0.741")
            changed = True
        # B1 No-effect: MLP=0.865 -> MLP=0.847
        if "No-effect" in item and "MLP=0.865" in item:
            item = item.replace("MLP=0.865", "MLP=0.847")
            changed = True
        if changed:
            q["ground_truth_key_facts"][i] = item
            ch.append(f"ground_truth_key_facts[{i}]: '{old_item}' -> '{item}'")
    changes_log["Q10"] = ch

    # =========================================================================
    # Q11
    # =========================================================================
    q = find_q(questions, "Q11")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "RF=0.885" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace("RF=0.885", "RF=0.884")
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q11"] = ch

    # =========================================================================
    # Q12 — COMPLETE REWRITE
    # =========================================================================
    q = find_q(questions, "Q12")
    ch = []
    # Preserve: id, modality, difficulty, category, data_context_files
    preserved = {k: q[k] for k in ["id", "modality", "category", "difficulty", "data_context_files"]}

    q["question"] = (
        "The B1 ablation shows that removing effect-size features actually improves XGBoost "
        "performance (0.911 \u2192 0.918 AUPRC) and barely affects RF (0.884 \u2192 0.863). What does "
        "this reveal about how different model families utilize feature types, and what implications "
        "does this have for feature engineering in genomic classification?"
    )
    ch.append("question: COMPLETE REWRITE (XGBoost crossover instead of MLP)")

    q["ground_truth_key_facts"] = [
        "B1 All features: XGBoost=0.911, RF=0.884, MLP=0.854",
        "B1 No-effect: XGBoost=0.918, RF=0.863, MLP=0.847",
        "B1 Effect-only: XGBoost=0.862, RF=0.813, MLP=0.741",
        "Removing effect-size features improves XGBoost by +0.007 but hurts RF by -0.021 and MLP by -0.007"
    ]
    ch.append("ground_truth_key_facts: COMPLETE REWRITE (4 new facts)")

    q["expected_reasoning"] = [
        "XGBoost may overfit to noisy effect-size features; removing them acts as implicit regularization",
        "RF relies more on effect-size features for splitting, so removal hurts performance",
        "Gradient boosting's sequential nature may amplify noise from correlated effect-size features",
        "Feature selection strategies should be model-aware \u2014 one-size-fits-all hurts some models"
    ]
    ch.append("expected_reasoning: COMPLETE REWRITE (4 new items)")

    q["rubric"]["factual_accuracy"] = (
        "Must correctly cite XGBoost improvement from 0.911 to 0.918 when effect-size features removed"
    )
    ch.append("rubric.factual_accuracy: REWRITTEN")

    # Verify preserved fields are intact
    for k, v in preserved.items():
        assert q[k] == v, f"Q12 field '{k}' was accidentally changed!"
    changes_log["Q12"] = ch

    # =========================================================================
    # Q15
    # =========================================================================
    q = find_q(questions, "Q15")
    ch = []
    for i, item in enumerate(q["expected_reasoning"]):
        if "8 PCA components" in item:
            old_val = item
            q["expected_reasoning"][i] = item.replace(
                "8 PCA components", "10 PCA components for C1 (8 per modality for G1)"
            )
            ch.append(f"expected_reasoning[{i}]: '{old_val}' -> '{q['expected_reasoning'][i]}'")
    changes_log["Q15"] = ch

    # =========================================================================
    # Q16
    # =========================================================================
    q = find_q(questions, "Q16")
    ch = []
    old_val = q["question"]
    q["question"] = q["question"].replace("barely above random (0.5)", "barely above random (0.529)")
    if q["question"] != old_val:
        ch.append(f"question: replaced 'barely above random (0.5)' -> 'barely above random (0.529)'")
    changes_log["Q16"] = ch

    # =========================================================================
    # Q18
    # =========================================================================
    q = find_q(questions, "Q18")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "C1 uses 8 PCA" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace("C1 uses 8 PCA", "C1 uses 10 PCA")
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q18"] = ch

    # =========================================================================
    # Q20
    # =========================================================================
    q = find_q(questions, "Q20")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "2,845 proteins shared" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace(
                "2,845 proteins shared",
                "380 proteins shared between biofluids (2,845 total plasma proteins)"
            )
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q20"] = ch

    # =========================================================================
    # Q28
    # =========================================================================
    q = find_q(questions, "Q28")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "21 main + 2 supplementary = 21" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace(
                "21 main + 2 supplementary = 21",
                "19 main + 2 supplementary = 21 total tasks"
            )
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q28"] = ch

    # =========================================================================
    # Q29
    # =========================================================================
    q = find_q(questions, "Q29")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "E4: best=0.023" in item:
            old_val = item
            # Replace 0.023 with 0.022 and also update the ratio if present
            new_item = item.replace("E4: best=0.023", "E4: best=0.022")
            # Also update any "8x random" to "7.3x random" since 0.022/0.003 = 7.3
            new_item = new_item.replace("8x random", "7.3x random")
            q["ground_truth_key_facts"][i] = new_item
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{new_item}'")
    changes_log["Q29"] = ch

    # =========================================================================
    # Q33
    # =========================================================================
    q = find_q(questions, "Q33")
    ch = []
    # question text: F2 "macro_f1=0.238" -> "macro_f1=0.280" and F5 "macro_f1=0.254" -> "macro_f1=0.304"
    old_val = q["question"]
    q["question"] = q["question"].replace("macro_f1=0.238", "macro_f1=0.280")
    q["question"] = q["question"].replace("macro_f1=0.254", "macro_f1=0.304")
    if q["question"] != old_val:
        ch.append(f"question: updated F2 and F5 scores")

    # ground_truth_key_facts: "F2: RF=0.238" -> "F2: LightGBM=0.280" and "F5: RF=0.254" -> "F5: LightGBM=0.304"
    for i, item in enumerate(q["ground_truth_key_facts"]):
        old_item = item
        changed = False
        if "F2: RF=0.238" in item:
            item = item.replace("F2: RF=0.238", "F2: LightGBM=0.280")
            changed = True
        if "F5: RF=0.254" in item:
            item = item.replace("F5: RF=0.254", "F5: LightGBM=0.304")
            changed = True
        if changed:
            q["ground_truth_key_facts"][i] = item
            ch.append(f"ground_truth_key_facts[{i}]: '{old_item}' -> '{item}'")
    changes_log["Q33"] = ch

    # =========================================================================
    # Q34
    # =========================================================================
    q = find_q(questions, "Q34")
    ch = []
    # question text: "F2=0.238, F5=0.254" -> "F2=0.280, F5=0.304"
    old_val = q["question"]
    q["question"] = q["question"].replace("F2=0.238", "F2=0.280")
    q["question"] = q["question"].replace("F5=0.254", "F5=0.304")
    if q["question"] != old_val:
        ch.append(f"question: updated F2 and F5 scores")

    # expected_reasoning: "3-class" -> "4-class"
    for i, item in enumerate(q["expected_reasoning"]):
        if "3-class" in item:
            old_val = item
            q["expected_reasoning"][i] = item.replace("3-class", "4-class")
            ch.append(f"expected_reasoning[{i}]: '{old_val}' -> '{q['expected_reasoning'][i]}'")
    changes_log["Q34"] = ch

    # =========================================================================
    # Q45
    # =========================================================================
    q = find_q(questions, "Q45")
    ch = []
    # I2 features: "4 aggregated features per pathway" -> "8 aggregated features per pathway"
    for i, item in enumerate(q["ground_truth_key_facts"]):
        old_item = item
        changed = False
        if "4 aggregated features per pathway" in item:
            item = item.replace("4 aggregated features per pathway", "8 aggregated features per pathway")
            changed = True
        if "4 aggregated features per gene" in item:
            item = item.replace("4 aggregated features per gene", "9 aggregated features per gene")
            changed = True
        if changed:
            q["ground_truth_key_facts"][i] = item
            ch.append(f"ground_truth_key_facts[{i}]: '{old_item}' -> '{item}'")

    # Add note about MLP tie for I3 best if LogReg=0.090 is mentioned
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "LogReg=0.090" in item and "I3" in item and "MLP" not in item:
            old_item = item
            q["ground_truth_key_facts"][i] = item.replace(
                "LogReg=0.090", "LogReg=0.090 (tied with MLP=0.090)"
            )
            ch.append(f"ground_truth_key_facts[{i}]: added MLP tie note: '{old_item}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q45"] = ch

    # =========================================================================
    # Q48
    # =========================================================================
    q = find_q(questions, "Q48")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "RF=0.005" in item and "I1" in item:
            old_item = item
            # Add LightGBM=0.006 (best) info
            if "LightGBM" not in item:
                q["ground_truth_key_facts"][i] = item.replace(
                    "RF=0.005", "RF=0.005, LightGBM=0.006 (best)"
                )
                ch.append(f"ground_truth_key_facts[{i}]: '{old_item}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q48"] = ch

    # =========================================================================
    # Q53
    # =========================================================================
    q = find_q(questions, "Q53")
    ch = []
    # In ground_truth_key_facts: where RF=0.266 is mentioned for H1, add "LightGBM=0.284 (overall best)"
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "RF=0.266" in item and "LightGBM=0.284" not in item:
            old_item = item
            q["ground_truth_key_facts"][i] = item.replace(
                "RF=0.266", "RF=0.266, LightGBM=0.284 (overall best)"
            )
            ch.append(f"ground_truth_key_facts[{i}]: '{old_item}' -> '{q['ground_truth_key_facts'][i]}'")

    # In question text: add LightGBM note if RF=0.266 is mentioned as if best
    old_val = q["question"]
    if "RF=0.266" in q["question"] and "LightGBM" not in q["question"]:
        # Replace "RF=0.266 AUPRC" with "RF=0.266 AUPRC (LightGBM=0.284 overall best)"
        q["question"] = q["question"].replace(
            "RF=0.266 AUPRC",
            "RF=0.266 AUPRC (LightGBM=0.284 overall best)"
        )
        # If that didn't match, try simpler replacement
        if q["question"] == old_val:
            q["question"] = q["question"].replace(
                "RF=0.266",
                "RF=0.266 (LightGBM=0.284 overall best)"
            )
    if q["question"] != old_val:
        ch.append(f"question: added LightGBM=0.284 note")

    # Add multi_omics.md to data_context_files if not present
    if "multi_omics.md" not in q["data_context_files"]:
        q["data_context_files"].append("multi_omics.md")
        ch.append("data_context_files: added 'multi_omics.md'")
    changes_log["Q53"] = ch

    # =========================================================================
    # Q64
    # =========================================================================
    q = find_q(questions, "Q64")
    ch = []
    # expected_reasoning and ground_truth_key_facts: RF=0.885 -> RF=0.884
    for i, item in enumerate(q["expected_reasoning"]):
        if "RF=0.885" in item or "0.885" in item:
            old_val = item
            q["expected_reasoning"][i] = item.replace("0.885", "0.884")
            ch.append(f"expected_reasoning[{i}]: '{old_val}' -> '{q['expected_reasoning'][i]}'")
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "RF=0.885" in item or "0.885" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace("0.885", "0.884")
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q64"] = ch

    # =========================================================================
    # Q77
    # =========================================================================
    q = find_q(questions, "Q77")
    ch = []
    # ground_truth_key_facts and question text: "LogReg=0.023" -> "LogReg=0.022" for E4
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "LogReg=0.023" in item or ("0.023" in item and "E4" in item):
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace("0.023", "0.022")
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    old_val = q["question"]
    # In question text, replace E4 0.023 with 0.022
    if "0.023" in q["question"]:
        q["question"] = q["question"].replace("0.023", "0.022")
        ch.append(f"question: replaced 0.023 -> 0.022")
    changes_log["Q77"] = ch

    # =========================================================================
    # Q89
    # =========================================================================
    q = find_q(questions, "Q89")
    ch = []
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "~40 features" in item:
            old_val = item
            q["ground_truth_key_facts"][i] = item.replace("~40 features", "39 features")
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q89"] = ch

    # =========================================================================
    # Q93
    # =========================================================================
    q = find_q(questions, "Q93")
    ch = []
    # expected_reasoning: remove F3 from LOCO tasks list (F3 uses LOTO, not LOCO)
    for i, item in enumerate(q["expected_reasoning"]):
        if "LOCO" in item and "F3" in item:
            old_val = item
            # Remove F3 from LOCO list - try various patterns
            new_item = item
            # Try patterns like "F1-F5" (which includes F3) -> "F1/F2/F4/F5"
            new_item = new_item.replace("F1-F5", "F1/F2/F4/F5")
            # Try "F3, " or ", F3" or "F3/" or "/F3"
            new_item = new_item.replace(", F3,", ",")
            new_item = new_item.replace(", F3 ", " ")
            new_item = new_item.replace("F3, ", "")
            new_item = new_item.replace(", F3", "")
            new_item = new_item.replace("/F3/", "/")
            new_item = new_item.replace("F3/", "")
            new_item = new_item.replace("/F3", "")
            if new_item != old_val:
                q["expected_reasoning"][i] = new_item
                ch.append(f"expected_reasoning[{i}]: '{old_val}' -> '{new_item}'")
    changes_log["Q93"] = ch

    # =========================================================================
    # Q98
    # =========================================================================
    q = find_q(questions, "Q98")
    ch = []
    for i, item in enumerate(q["expected_reasoning"]):
        if "0.912 XGB" in item or "0.912" in item:
            old_val = item
            q["expected_reasoning"][i] = item.replace("0.912", "0.911")
            ch.append(f"expected_reasoning[{i}]: '{old_val}' -> '{q['expected_reasoning'][i]}'")
    changes_log["Q98"] = ch

    # =========================================================================
    # Q100
    # =========================================================================
    q = find_q(questions, "Q100")
    ch = []
    # Fix ground_truth_key_facts: task win counts
    for i, item in enumerate(q["ground_truth_key_facts"]):
        if "Task wins" in item or ("LightGBM=" in item and "LogReg=" in item and "RF=" in item and "MLP=" in item):
            old_val = item
            q["ground_truth_key_facts"][i] = (
                "Task wins (19 main tasks): LightGBM=8, LogReg=7 (incl. B2), RF=2, MLP=2 (C1, I3), XGBoost=0"
            )
            ch.append(f"ground_truth_key_facts[{i}]: '{old_val}' -> '{q['ground_truth_key_facts'][i]}'")
    changes_log["Q100"] = ch

    # =========================================================================
    # SAVE
    # =========================================================================
    save_json(data, INPUT)

    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("=" * 80)
    print("QUESTION BANK FIX SUMMARY")
    print("=" * 80)
    total_changes = 0
    for qid in sorted(changes_log.keys(), key=lambda x: int(x[1:])):
        changes = changes_log[qid]
        if changes:
            print(f"\n--- {qid} ({len(changes)} change(s)) ---")
            for c in changes:
                print(f"  * {c}")
            total_changes += len(changes)
        else:
            print(f"\n--- {qid} (NO CHANGES - check data!) ---")

    print(f"\n{'=' * 80}")
    print(f"TOTAL: {total_changes} changes across {len(changes_log)} questions")
    print(f"{'=' * 80}")

    # =========================================================================
    # VERIFY: re-read and spot-check
    # =========================================================================
    print("\n\nVERIFICATION SPOT CHECKS:")
    print("-" * 40)
    data2 = load_json(INPUT)
    qs = {q["id"]: q for q in data2["questions"]}

    checks = [
        ("Q01", "expected_reasoning[0]", lambda: "39 features (20 CBC + 19 CMP)" in qs["Q01"]["expected_reasoning"][0]),
        ("Q01", "rubric.factual_accuracy", lambda: "39 features (20 CBC + 19 CMP)" in qs["Q01"]["rubric"]["factual_accuracy"]),
        ("Q02", "rubric.factual_accuracy N=28", lambda: "N=28" in qs["Q02"]["rubric"]["factual_accuracy"]),
        ("Q03", "ground_truth_key_facts CBC+CMP", lambda: any("CBC + CMP, 39 features" in x for x in qs["Q03"]["ground_truth_key_facts"])),
        ("Q04", "expected_reasoning 28 samples", lambda: any("28 samples" in x for x in qs["Q04"]["expected_reasoning"])),
        ("Q06", "ground_truth_key_facts N=28", lambda: any("N=28 total" in x for x in qs["Q06"]["ground_truth_key_facts"])),
        ("Q07", "ground_truth_key_facts 28 samples", lambda: any("28 samples" in x for x in qs["Q07"]["ground_truth_key_facts"])),
        ("Q10", "question 0.884", lambda: "0.884" in qs["Q10"]["question"]),
        ("Q10", "MLP=0.854 in All", lambda: any("MLP=0.854" in x and "All" in x for x in qs["Q10"]["ground_truth_key_facts"])),
        ("Q10", "MLP=0.741 in Effect-only", lambda: any("MLP=0.741" in x and "Effect" in x for x in qs["Q10"]["ground_truth_key_facts"])),
        ("Q10", "MLP=0.847 in No-effect", lambda: any("MLP=0.847" in x and "No-effect" in x for x in qs["Q10"]["ground_truth_key_facts"])),
        ("Q11", "RF=0.884", lambda: any("RF=0.884" in x for x in qs["Q11"]["ground_truth_key_facts"])),
        ("Q12", "XGBoost in question", lambda: "XGBoost" in qs["Q12"]["question"]),
        ("Q12", "0.911 in question", lambda: "0.911" in qs["Q12"]["question"]),
        ("Q12", "preserved id", lambda: qs["Q12"]["id"] == "Q12"),
        ("Q12", "preserved modality", lambda: qs["Q12"]["modality"] == "transcriptomics"),
        ("Q15", "10 PCA", lambda: any("10 PCA" in x for x in qs["Q15"]["expected_reasoning"])),
        ("Q16", "0.529", lambda: "0.529" in qs["Q16"]["question"]),
        ("Q18", "C1 uses 10 PCA", lambda: any("C1 uses 10 PCA" in x for x in qs["Q18"]["ground_truth_key_facts"])),
        ("Q20", "380 proteins", lambda: any("380 proteins" in x for x in qs["Q20"]["ground_truth_key_facts"])),
        ("Q28", "19 main", lambda: any("19 main" in x for x in qs["Q28"]["ground_truth_key_facts"])),
        ("Q29", "E4: best=0.022", lambda: any("best=0.022" in x for x in qs["Q29"]["ground_truth_key_facts"])),
        ("Q33", "F2: LightGBM=0.280", lambda: any("LightGBM=0.280" in x for x in qs["Q33"]["ground_truth_key_facts"])),
        ("Q33", "F5: LightGBM=0.304", lambda: any("LightGBM=0.304" in x for x in qs["Q33"]["ground_truth_key_facts"])),
        ("Q34", "F2=0.280 in question", lambda: "F2=0.280" in qs["Q34"]["question"]),
        ("Q34", "4-class", lambda: any("4-class" in x for x in qs["Q34"]["expected_reasoning"])),
        ("Q45", "8 aggregated per pathway", lambda: any("8 aggregated features per pathway" in x for x in qs["Q45"]["ground_truth_key_facts"])),
        ("Q45", "9 aggregated per gene", lambda: any("9 aggregated features per gene" in x for x in qs["Q45"]["ground_truth_key_facts"])),
        ("Q48", "LightGBM=0.006", lambda: any("LightGBM=0.006" in x for x in qs["Q48"]["ground_truth_key_facts"])),
        ("Q53", "LightGBM=0.284", lambda: any("LightGBM=0.284" in x for x in qs["Q53"]["ground_truth_key_facts"])),
        ("Q64", "RF=0.884", lambda: any("0.884" in x for x in qs["Q64"]["ground_truth_key_facts"])),
        ("Q77", "0.022 in question", lambda: "0.022" in qs["Q77"]["question"]),
        ("Q89", "39 features", lambda: any("39 features" in x for x in qs["Q89"]["ground_truth_key_facts"])),
        ("Q93", "F3 not in LOCO", lambda: not any("LOCO" in x and "F3" in x for x in qs["Q93"]["expected_reasoning"])),
        ("Q98", "0.911 not 0.912", lambda: not any("0.912" in x for x in qs["Q98"]["expected_reasoning"])),
        ("Q100", "LightGBM=8", lambda: any("LightGBM=8" in x for x in qs["Q100"]["ground_truth_key_facts"])),
        ("Q100", "LogReg=7", lambda: any("LogReg=7" in x for x in qs["Q100"]["ground_truth_key_facts"])),
        ("Q100", "MLP=2", lambda: any("MLP=2" in x for x in qs["Q100"]["ground_truth_key_facts"])),
    ]

    passed = 0
    failed = 0
    for qid, desc, check in checks:
        result = check()
        status = "PASS" if result else "FAIL"
        if result:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {qid}: {desc}")

    print(f"\nVerification: {passed}/{passed+failed} checks passed")
    if failed > 0:
        print("WARNING: Some checks failed! Review the output above.")
        sys.exit(1)
    else:
        print("All checks passed!")

if __name__ == "__main__":
    main()
