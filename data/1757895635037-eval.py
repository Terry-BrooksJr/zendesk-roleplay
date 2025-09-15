#!/usr/bin/env python3
import argparse, json, os, time, statistics
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from datasets import load_dataset, Dataset
from datetime import timezone
from evaluate import load as load_metric
from transformers import pipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_splits(cfg: dict) -> dict:
    data_files = {}
    ds_cfg = cfg["datasets"]
    for split in ("train", "validation", "test"):
        if split in ds_cfg and ds_cfg[split]:
            data_files[split] = ds_cfg[split]
    if not data_files:
        raise ValueError("No dataset files configured in usecase.yaml")
    return load_dataset("json", data_files=data_files)


def to_label_index_map(labels):
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def confusion_matrix(y_true_idx, y_pred_idx, n_classes):
    mat = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_idx, y_pred_idx):
        mat[t, p] += 1
    return mat


def per_class_f1(y_true, y_pred, labels):
    f1_metric = load_metric("f1")
    scores = {}
    for lab in labels:
        y_true_bin = [1 if y == lab else 0 for y in y_true]
        y_pred_bin = [1 if y == lab else 0 for y in y_pred]
        scores[lab] = f1_metric.compute(predictions=y_pred_bin, references=y_true_bin, average="binary")["f1"]
    return scores


def build_pipe(model_spec: dict, device="auto"):
    m = model_spec["model"]
    t = model_spec.get("type", None)
    # If type not provided, infer a sensible default
    if not t:
        if "mnli" in m.lower() or "bart-large-mnli" in m.lower():
            t = "zero-shot-classification"
        else:
            t = "text-classification"
    return pipeline(task=t, model=m, device_map=device, truncation=True, top_k=None)


def eval_model_on_split(pipe, split: Dataset, labels, text_col: str, label_col: str, task_type: str):
    acc_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")

    preds, refs, latencies = [], [], []
    for row in split:
        text = row[text_col]
        y_true = row[label_col]

        start = time.perf_counter()
        if task_type == "zero-shot-classification":
            out = pipe(text, candidate_labels=labels, multi_label=True)
            # top label is rank 0
            y_pred = out["labels"][0] if isinstance(out, dict) else out[0]["labels"][0]
        else:
            out = pipe(text)
            # handle list vs dict, and LABEL_X vs real label names
            if isinstance(out, list) and out:
                pred = out[0]
            elif isinstance(out, dict):
                pred = out
            else:
                raise RuntimeError("Unexpected pipeline output format")
            # normalize
            y_pred = pred.get("label") or pred.get("labels", [""])[0]
        latency = time.perf_counter() - start

        preds.append(y_pred)
        refs.append(y_true)
        latencies.append(latency)

    # metrics
    acc = acc_metric.compute(predictions=preds, references=refs)["accuracy"]
    f1_macro = f1_metric.compute(predictions=preds, references=refs, average="macro")["f1"]

    # per-class F1
    f1_by_class = per_class_f1(refs, preds, labels)

    # latency stats
    p50 = statistics.median(latencies)
    p95 = float(np.percentile(latencies, 95))

    # confusion matrix
    label2id, _ = to_label_index_map(labels)
    y_true_idx = [label2id.get(y, -1) for y in refs]
    y_pred_idx = [label2id.get(y, -1) for y in preds]
    # filter any -1 (unknown labels)
    if (filt := [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if t != -1 and p != -1]):
        y_true_idx, y_pred_idx = zip(*filt)
        cm = confusion_matrix(y_true_idx, y_pred_idx, len(labels)).tolist()
    else:
        cm = [[0]*len(labels) for _ in labels]

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_by_class": f1_by_class,
        "latency_p50": p50,
        "latency_p95": p95,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm
        },
        "num_examples": len(refs)
    }


def evaluate_all(cfg: dict):
    labels = cfg["labels"]
    text_col = cfg.get("evaluation", {}).get("text_column", "text")
    label_col = cfg.get("evaluation", {}).get("label_column", "label")
    split_name = cfg.get("evaluation", {}).get("split", "validation")
    device = cfg.get("constraints", {}).get("device", "auto")

    ds = load_splits(cfg)
    if split_name not in ds:
        raise ValueError(f"Split '{split_name}' not found in datasets. Available: {list(ds.keys())}")

    results = []
    for cand in cfg.get("candidates", []):
        pipe = build_pipe(cand, device=device)
        r = eval_model_on_split(
            pipe,
            ds[split_name],
            labels=labels,
            text_col=text_col,
            label_col=label_col,
            task_type=cand.get("type", pipe.task)
        )
        model_id = cand["model"]
        results.append({"model": model_id, **r})

    return results


def check_thresholds(cfg, model_result):
    thresholds = {m["name"]: m for m in cfg.get("metrics", [])}
    verdicts = []
    for key, val in (("accuracy", model_result["accuracy"]), ("f1_macro", model_result["f1_macro"]), ("latency_p95", model_result["latency_p95"])):
        if key in thresholds and "threshold" in thresholds[key]:
            thr = thresholds[key]["threshold"]
            ok = (val >= thr) if key != "latency_p95" else (val <= thr)
            verdicts.append({"metric": key, "value": val, "threshold": thr, "pass": ok})
    overall = all(v["pass"] for v in verdicts) if verdicts else True
    return {"checks": verdicts, "overall_pass": overall}


def write_reports(cfg, results):
    out_dir = Path(cfg.get("report", {}).get("output_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # JSON
    json_report = {
        "use_case": cfg.get("use_case"),
        "task": cfg.get("task"),
        "labels": cfg.get("labels"),
        "split": cfg.get("evaluation", {}).get("split", "validation"),
        "results": []
    }
    for r in results:
        entry = dict(r)
        entry["thresholds"] = check_thresholds(cfg, r)
        json_report["results"].append(entry)

    json_path = out_dir / f"report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    # Markdown
    md_lines = [f"# Evaluation Report — {cfg.get('use_case', '(unnamed)')}"]
    md_lines.append(f"- **Task**: {cfg.get('task')}")
    md_lines.extend((f"- **Split**: {json_report['split']}", ""))
    for r in json_report["results"]:
        md_lines.extend(
            (
                f"## Model: `{r['model']}`",
                f"- Accuracy: **{r['accuracy']:.3f}**",
                f"- F1 (macro): **{r['f1_macro']:.3f}**",
                f"- Latency p50: **{r['latency_p50'] * 1000:.1f} ms**",
                f"- Latency p95: **{r['latency_p95'] * 1000:.1f} ms**",
            )
        )
        if checks := r["thresholds"]["checks"]:
            md_lines.append("- Threshold checks:")
            for c in checks:
                status = "✅" if c["pass"] else "❌"
                md_lines.append(f"  - {status} {c['metric']}: {c['value']:.3f} vs threshold {c['threshold']}")
        # per-class F1
        md_lines.append("\nPer-class F1:")
        df = pd.DataFrame.from_dict(r["f1_by_class"], orient="index", columns=["f1"]).sort_values("f1", ascending=False)
        md_lines.extend((df.to_markdown(), "\nConfusion Matrix:"))
        labs = r["confusion_matrix"]["labels"]
        mat = r["confusion_matrix"]["matrix"]
        df_cm = pd.DataFrame(mat, index=[f"T:{l}" for l in labs], columns=[f"P:{l}" for l in labs])
        md_lines.extend((df_cm.to_markdown(), "\n---\n"))
    md_path = out_dir / f"report_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    return str(json_path), str(md_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="usecase.yaml", help="Path to usecase.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = evaluate_all(cfg)
    json_path, md_path = write_reports(cfg, results)

    print(f"✅ Evaluation complete.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()