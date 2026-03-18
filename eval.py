

import os, sys, json, csv, argparse, math, random
from typing import Dict, Tuple, List
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset, get_dataset_config_names
import evaluate

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    ElectraForPreTraining,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)


from torch.optim import AdamW


# ---------------- CONFIG ----------------
GLUE_TASKS = ["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# primary num labels (others default to 2)
TASK_NUM_LABELS = {"stsb": 1, "mnli": 3}

# conservative per-task max sequence lengths (speed!)
TASK_MAXLEN = {
    "cola": 128, "sst2": 128, "mrpc": 128, "qqp": 128,
    "stsb": 128, "mnli": 128, "qnli": 128, "rte": 128, "wnli": 128,
}

# Task-specific training recipes (helpful for CoLA)
TASK_HPARAMS = {
    # epochs, lr, warmup_ratio, weight_decay, batch_size
    "cola":  {"epochs": 6, "lr": 2e-5, "warmup_ratio": 0.06, "wd": 0.01, "batch_size": 32},
    "sst2":  {"epochs": 3, "lr": 2e-5, "warmup_ratio": 0.06, "wd": 0.01, "batch_size": 32},
    "mrpc":  {"epochs": 5, "lr": 2e-5, "warmup_ratio": 0.1,  "wd": 0.01, "batch_size": 32},
    "qqp":   {"epochs": 3, "lr": 2e-5, "warmup_ratio": 0.06, "wd": 0.01, "batch_size": 64},
    "stsb":  {"epochs": 5, "lr": 2e-5, "warmup_ratio": 0.06, "wd": 0.01, "batch_size": 32},
    "mnli":  {"epochs": 3, "lr": 2e-5, "warmup_ratio": 0.06, "wd": 0.01, "batch_size": 64},
    "qnli":  {"epochs": 3, "lr": 2e-5, "warmup_ratio": 0.06, "wd": 0.01, "batch_size": 32},
    "rte":   {"epochs": 8, "lr": 2e-5, "warmup_ratio": 0.1,  "wd": 0.01, "batch_size": 32},
    "wnli":  {"epochs":10, "lr": 2e-5, "warmup_ratio": 0.1,  "wd": 0.01, "batch_size": 32},
}


# ---------------- UTILITIES ----------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- TOKENIZER LOADING ----------------
def load_tokenizer(tokenizer_dir: str = None, vocab_path: str = None, do_lower_case: bool = None):
    """
    Load a HF tokenizer. Prefer `tokenizer_dir` (expects vocab + configs).
    If only a vocab.txt is available, build a BertTokenizerFast around it.

    NOTE: We do NOT force lowercasing by default. Pass --do_lower_case if your vocab is uncased.
    """
    if tokenizer_dir:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
        print("**************************************************************")
        print(f"[tokenizer] loaded from checkpoint dir: {tokenizer_dir}")
        return tok

    if vocab_path:
        print('#####################################')
        from transformers import BertTokenizerFast
        from transformers import ElectraTokenizerFast

        extra = {}
        if do_lower_case is not None:
            extra["do_lower_case"] = do_lower_case
        tok = BertTokenizerFast(
            vocab_file=vocab_path,
            cls_token='[CLS]', sep_token='[SEP]',
            pad_token='[PAD]', mask_token='[MASK]', unk_token='[UNK]',
            **extra,
        )
           # tok = BertTokenizerFast(
        #     vocab_file=vocab_path,
        #     cls_token='[CLS]', sep_token='[SEP]',
        #     pad_token='[PAD]', mask_token='[MASK]', unk_token='[UNK]',
        #     **extra,
        # )
        # tok = ElectraTokenizerFast(
        #     vocab_file=vocab_path,
        #     cls_token='[CLS]', sep_token='[SEP]',
        #     pad_token='[PAD]', mask_token='[MASK]', unk_token='[UNK]',
        #     **extra,
        # )
        # print(f"[tokenizer] loaded from vocab: {vocab_path} (do_lower_case={extra.get('do_lower_case', None)})")
        return tok

    raise RuntimeError("No tokenizer could be loaded. Provide --tokenizer_dir or --vocab_path.")


# ---------------- DATALOADER ----------------
def build_dataloader(raw_ds, tokenizer, task, batch_size, shuffle, num_workers=4) -> DataLoader:
    s1, s2 = TASK_TO_KEYS[task]
    max_len = TASK_MAXLEN.get(task, 128)

    def preprocess(examples):
        texts = (examples[s1],) if s2 is None else (examples[s1], examples[s2])
        out = tokenizer(*texts, truncation=True, max_length=max_len)
        if "label" in examples:
            out["labels"] = examples["label"]
        return out

    ds = raw_ds.map(preprocess, batched=True, remove_columns=raw_ds.column_names)
    coll = DataCollatorWithPadding(tokenizer)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=coll,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# ---------------- ELECTRA DISC HELPER (BLiMP) ----------------
@torch.no_grad()
def electra_disc_sentence_score(disc, tok, sent, device, max_length=128):
    enc = tok(sent, return_tensors="pt", truncation=True, max_length=max_length)
    ids, attn = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    out = disc(input_ids=ids, attention_mask=attn)
    p_replaced = torch.sigmoid(out.logits[0])
    score = 0.0
    for i in range(ids.size(1)):
        if attn[0, i] == 0:
            continue
        tid = ids[0, i].item()
        if hasattr(tok, "all_special_ids") and tid in tok.all_special_ids:
            continue
        score += torch.log1p(-p_replaced[i].clamp(1e-6, 1 - 1e-6)).item()
    return score


# ---------------- FINE-TUNE + EVAL (GLUE) ----------------
def finetune_and_eval_glue(
    base_model_dir: str,
    tokenizer,
    task: str,
    out_dir: str,
    device: torch.device,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    warmup_ratio: float = None,
    weight_decay: float = None,
    grad_accum: int = 2,
    fp16: bool = True,
    max_train_samples: int = None,
    max_eval_samples: int = None,
    max_steps_per_epoch: int = None,
) -> Dict[str, float]:

    # Merge defaults with task-specific recipe
    hp = TASK_HPARAMS[task].copy()
    if epochs is not None:       hp["epochs"] = epochs
    if batch_size is not None:   hp["batch_size"] = batch_size
    if lr is not None:           hp["lr"] = lr
    if warmup_ratio is not None: hp["warmup_ratio"] = warmup_ratio
    if weight_decay is not None: hp["wd"] = weight_decay

    os.makedirs(out_dir, exist_ok=True)

    cfg = AutoConfig.from_pretrained(base_model_dir)
    cfg.num_labels = TASK_NUM_LABELS.get(task, 2)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir, config=cfg, ignore_mismatched_sizes=True
    ).to(device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # dataset
    ds = load_dataset("glue", task)
    if max_train_samples:
        ds["train"] = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))

    val_main_split = "validation_matched" if task == "mnli" else "validation"
    if max_eval_samples:
        ds[val_main_split] = ds[val_main_split].select(range(min(max_eval_samples, len(ds[val_main_split]))))
        if task == "mnli":
            ds["validation_mismatched"] = ds["validation_mismatched"].select(
                range(min(max_eval_samples, len(ds["validation_mismatched"])))
            )

    train_dl = build_dataloader(ds["train"], tokenizer, task, hp["batch_size"], shuffle=True)
    val_main_dl = build_dataloader(ds[val_main_split], tokenizer, task, hp["batch_size"], shuffle=False)
    val_mm_dl = build_dataloader(ds["validation_mismatched"], tokenizer, task, hp["batch_size"], shuffle=False) \
        if task == "mnli" else None

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    total_train_steps = (len(train_dl) // max(1, grad_accum)) * hp["epochs"]
    warmup_steps = int(hp["warmup_ratio"] * total_train_steps)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
    )

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device.type == "cuda"))
    autocast = torch.cuda.amp.autocast if scaler.is_enabled() else nullcontext

    # ---- TRAIN ----
    metric = evaluate.load("glue", task)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    try:
        for ep in range(hp["epochs"]):
            print(f"[train {task}] epoch {ep+1}/{hp['epochs']}  (bs={hp['batch_size']}, lr={hp['lr']}, warmup={hp['warmup_ratio']}, wd={hp['wd']})", flush=True)
            step_in_epoch = 0
            for i, batch in enumerate(tqdm(train_dl, leave=False)):
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast():
                    out = model(**batch)
                    loss = out.loss / max(1, grad_accum)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % grad_accum == 0:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                step_in_epoch += 1
                if max_steps_per_epoch and step_in_epoch >= max_steps_per_epoch:
                    break
    except KeyboardInterrupt:
        print(f"\n[warn] KeyboardInterrupt during {task} – saving partial checkpoint…", flush=True)

    # save finetuned model
    try:
        model.save_pretrained(out_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out_dir)
    except Exception:
        pass

    # ---- EVAL (main) ----
    model.eval()
    for batch in tqdm(val_main_dl, desc=f"eval {task}", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        preds = out.logits.argmax(-1) if task != "stsb" else out.logits.squeeze(-1)
        metric.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
    res = metric.compute()

    # ---- EVAL (MNLI mismatched) ----
    if val_mm_dl is not None:
        metric_mm = evaluate.load("glue", "mnli_mismatched")
        for batch in tqdm(val_mm_dl, desc=f"eval {task}-mm", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                out = model(**batch)
            preds = out.logits.argmax(-1)
            metric_mm.add_batch(predictions=preds.cpu(), references=batch["labels"].cpu())
        res_mm = metric_mm.compute()
        res = {"mnli_matched": res, "mnli_mismatched": res_mm}

    print(f"[eval {task}] {res}", flush=True)
    return res


# ---------------- BLiMP (ELECTRA DISCRIMINATOR) ----------------
def run_blimp_electra(model_dir: str, tok, device: torch.device) -> Dict[str, float]:
    disc = ElectraForPreTraining.from_pretrained(model_dir).to(device).eval()
    results = {}
    configs = get_dataset_config_names("blimp")

    total_num, total_den = 0.0, 0
    for conf in configs:
        ds = load_dataset("blimp", conf)["train"]
        corr, tot = 0, 0
        for ex in tqdm(ds, desc=f"BLiMP {conf}", leave=False):
            g = electra_disc_sentence_score(disc, tok, ex["sentence_good"], device)
            b = electra_disc_sentence_score(disc, tok, ex["sentence_bad"], device)
            corr += 1 if g > b else 0
            tot += 1
        acc = corr / max(tot, 1)
        results[conf] = acc
        total_num += acc * tot
        total_den += tot

    results["_overall_accuracy"] = total_num / max(total_den, 1)
    return results


# ---------------- SUMMARY & REPORTS ----------------
def summarize_glue(all_glue: dict) -> Tuple[List[Tuple[str, float, str]], float]:
    """Flatten metrics + compute primary scores and GLUE score."""
    def safe(d, k, default=0.0): return d.get(k, default)

    rows, primaries = [], []
    for task in GLUE_TASKS:
        m = all_glue.get(task, {})

        if task == "mnli" and isinstance(m, dict) and "mnli_matched" in m:
            m_main = m["mnli_matched"]; m_mm = m.get("mnli_mismatched", {})
            primary = safe(m_main, "accuracy")
            metrics_str = "matched_acc={:.4f}, mismatched_acc={:.4f}".format(
                safe(m_main, "accuracy"), safe(m_mm, "accuracy"))
            rows.append((task.upper(), primary, metrics_str))
            primaries.append(primary)
            continue

        if task == "cola":
            primary = safe(m, "matthews_correlation")
        elif task in ("sst2", "qnli", "rte", "wnli"):
            primary = safe(m, "accuracy")
        elif task in ("mrpc", "qqp"):
            primary = (safe(m, "accuracy") + safe(m, "f1")) / 2.0
        elif task == "stsb":
            primary = (safe(m, "pearson") + safe(m, "spearmanr")) / 2.0
        else:
            primary = safe(m, "accuracy")

        primaries.append(primary)
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()))
        rows.append((task.upper(), primary, metrics_str if metrics_str else "-"))

    glue_score = sum(primaries) / max(len(primaries), 1)
    return rows, glue_score


def write_reports(output_root: str, glue_rows, glue_score, blimp_res):
    os.makedirs(output_root, exist_ok=True)

    # Pretty text table
    col1, col2, col3 = "TASK", "PRIMARY", "ALL METRICS"
    w1 = max(len(col1), max(len(r[0]) for r in glue_rows))
    w2 = max(len(col2), 8)
    header = f"{col1:<{w1}}  {col2:>{w2}}  {col3}"
    lines = [header, "-" * len(header)]
    for t, p, ms in glue_rows:
        lines.append(f"{t:<{w1}}  {p:>{w2}.4f}  {ms}")
    lines.append("-" * len(header))
    lines.append(f"{'GLUE SCORE':<{w1}}  {glue_score:>{w2}.4f}  (mean of task primaries)")
    if "_overall_accuracy" in blimp_res:
        lines.append(f"{'BLiMP OVERALL':<{w1}}  {blimp_res['_overall_accuracy']:>{w2}.4f}  (disc preference)")

    # # Save text
    # txt_path = os.path.join(output_root, "final_summary.txt")
    # with open(txt_path, "w") as f:
    #     f.write("\n".join(lines))
    # print("\n" + "\n".join(lines))
    # print(f"\n[summary] wrote {txt_path}")

    # CSV
    csv_path = os.path.join(output_root, "final_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "primary_metric", "metrics"])
        for t, p, ms in glue_rows:
            writer.writerow([t, f"{p:.6f}", ms])
        writer.writerow([])
        writer.writerow(["GLUE_SCORE", f"{glue_score:.6f}"])
        if "_overall_accuracy" in blimp_res:
            writer.writerow(["BLIMP_OVERALL", f"{blimp_res['_overall_accuracy']:.6f}"])
    print(f"[summary] wrote {csv_path}")
    ##### To enable markdown output resultls
    # # Markdown ##
    # md_path = os.path.join(output_root, "final_summary.md")
    # with open(md_path, "w") as f:
    #     f.write("| Task | Primary | All Metrics |\n|---|---:|---|\n")
    #     for t, p, ms in glue_rows:
    #         f.write(f"| {t} | {p:.4f} | {ms} |\n")
    #     f.write(f"\n**GLUE Score:** {glue_score:.4f}\n\n")
    #     if "_overall_accuracy" in blimp_res:
    #         f.write(f"**BLiMP Overall:** {blimp_res['_overall_accuracy']:.4f} (discriminator preference)\n")
    # print(f"[summary] wrote {md_path}")


# ---------------- ARGS & MAIN ----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune ELECTRA on GLUE and evaluate BLiMP. Writes reports to output dir."
    )
    ap.add_argument("--model", required=True,
                    help="Path to pretrained checkpoint dir (contains config.json + model weights).")
    ap.add_argument("--output_root", default="./outputs_electra_eval",
                    help="Output directory for checkpoints and reports.")

    # Tokenizer options (provide at least one)
    ap.add_argument("--tokenizer_dir", default=None,
                    help="Directory with tokenizer files (vocab/tokenizer_config/special_tokens_map).")
    ap.add_argument("--vocab_path", default=None,
                    help="Path to vocab.txt if tokenizer_dir is not provided.")
    ap.add_argument("--do_lower_case", action="store_true",
                    help="Set if your vocab/model was trained uncased (BERT/ELECTRA uncased).")

    # Training hyperparams (global defaults that can be overridden per task)
    ap.add_argument("--epochs", type=int, default=None, help="Override task recipe epochs")
    ap.add_argument("--batch_size", type=int, default=None, help="Override task recipe batch size")
    ap.add_argument("--lr", type=float, default=None, help="Override task recipe LR")
    ap.add_argument("--warmup_ratio", type=float, default=None, help="Override task recipe warmup ratio")
    ap.add_argument("--weight_decay", type=float, default=None, help="Override task recipe weight decay")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--fp16", action="store_true", default=True)

    # Speed/size caps
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="Cap training examples per task (e.g., 60000 for QQP).")
    ap.add_argument("--max_eval_samples", type=int, default=None,
                    help="Cap validation examples per task.")
    ap.add_argument("--max_steps_per_epoch", type=int, default=None,
                    help="Limit steps each epoch (e.g., 1200) — handy for QQP/MNLI.")

    # Task selection
    ap.add_argument("--tasks", type=str, default="all",
                    help="Comma-separated subset like 'sst2,qnli' or 'all'.")

    # BLiMP
    ap.add_argument("--blimp_model_dir", default=None,
                    help="(Optional) Evaluate BLiMP on this model instead of base.")

    # Repro
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch={torch.__version__}  device={device}", flush=True)
    print(f"[paths] model={args.model}", flush=True)
    print(f"[paths] output_root={args.output_root}", flush=True)

    tokenizer = load_tokenizer(args.tokenizer_dir, args.vocab_path,
                               do_lower_case=(True if args.do_lower_case else None))

    # ---- Fine-tune + eval GLUE ----
    if args.tasks.lower() == "all":
        tasks = GLUE_TASKS
    else:
        tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]
        for t in tasks:
            if t not in GLUE_TASKS:
                raise ValueError(f"Unknown task '{t}'. Allowed: {GLUE_TASKS}")

    all_glue = {}
    for task in tasks:
        out_dir = os.path.join(args.output_root, f"electra_{task}_ft")
        res = finetune_and_eval_glue(
            args.model, tokenizer, task, out_dir, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
            grad_accum=args.grad_accum, fp16=args.fp16,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            max_steps_per_epoch=(args.max_steps_per_epoch if task in ("qqp","mnli") else None),
        )

        # Flatten MNLI result for storage
        if task == "mnli" and isinstance(res, dict) and "mnli_matched" in res:
            all_glue[task] = {
                "accuracy": res["mnli_matched"].get("accuracy", 0.0),
                "accuracy_mm": res["mnli_mismatched"].get("accuracy", 0.0)
            }
        else:
            all_glue[task] = res

    glue_json = os.path.join(args.output_root, "all_glue_results.json")
    with open(glue_json, "w") as f:
        json.dump(all_glue, f, indent=2)
    print(f"[GLUE] saved -> {glue_json}", flush=True)

    # ---- BLiMP ----
    blimp_model = args.blimp_model_dir or args.model
    print(f"\n[BLiMP] evaluating ELECTRA discriminator at: {blimp_model}", flush=True)
    blimp_res = run_blimp_electra(blimp_model, tokenizer, device)
    blimp_json = os.path.join(args.output_root, "blimp_results.json")
    with open(blimp_json, "w") as f:
        json.dump(blimp_res, f, indent=2)
    print(f"[BLiMP] saved -> {blimp_json}", flush=True)

    # ---- Reports ----
    glue_rows, glue_score = summarize_glue(all_glue)
    write_reports(args.output_root, glue_rows, glue_score, blimp_res)
    print("\n[done] All reports written under:", args.output_root)


if __name__ == "__main__":
    main()
# TOKENIZERS_PARALLELISM=false python -u eval.py \
#   --model /home/osamanatouf/ELECTR_Final_submission/electra-pytorch/output/pretrain/2025-10-27-18-27-50_250000_steps_recheck_res/ckpt/250000\
#   --vocab_path /home/osamanatouf/CS557_final_project/electra-pytorch/data/vocab.txt \
#   --output_root ./aoa_250_recheck \
#   --do_lower_case --fp16 --grad_accum 2 --tasks all

# TOKENIZERS_PARALLELISM=false python -u eval.py \
#   --model /home/osamanatouf/CS557_final_project/electra-pytorch/output/pretrain_aoa/2025-11-08-17-21-38modified_scheduler_250000_steps_all_train_after_6th_bin_with_small_precetange_allowed_during_from_current_bin_during_freeze_0.05/ckpt/250000\
#   --vocab_path /home/osamanatouf/CS557_final_project/electra-pytorch/data/vocab.txt \
#   --output_root ./outputs_electra_eval_250K_AOA_CONFIRM_retrained_again_to_check_res_with_soft_freezing_0.05 \
#   --do_lower_case --fp16 --grad_accum 2 --tasks all
# 250K step, is the flexable training with the model and masking input since the token level did not have good mapping.
#/home/osamanatouf/ELECTR_Final_submission/electra-pytorch/output/pretrain_aoa/2025-10-23-17-47-39modified_scheduler_250000_steps_all_train_after_6th_bin_with_small_precetange_allowed_during_from_current_bin_during_freeze/ckpt/250000

# TOKENIZERS_PARALLELISM=false python -m electra-pytorch.eval   --model //home/osamanatouf/CS557_final_project/electra-pytorch/output/pretrain/2025-10-25-14-31-19_350000_steps/ckpt/350000  --vocab_path /home/osamanatouf/CS557_final_project/electra-pytorch/data/vocab.txt   --output_root ./outputs_electra_eval_350K_BASE_NO_AOA   --do_lower_case --fp16 --grad_accum 2 --tasks all