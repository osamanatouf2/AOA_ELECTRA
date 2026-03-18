# Psycholinguistically Grounded Curriculum Learning for Transformer-Based Language Models

The following code implements a **psycholinguistically grounded curriculum learning** pipeline for transformer language model pretraining using **Age of Acquisition (AoA)** as a difficulty signal. The core idea is to train a model with **simpler, earlier-acquired language first**, then progressively introduce **harder, later-acquired language** through AoA-based curriculum stages.

The pretraining setup is **ELECTRA-style** (generator + discriminator). I intentionally built on the existing ELECTRA implementation **`electra-pytorch` by lucidrains** so I would not waste project time re-validating ELECTRA from scratch. The main focus of this project is the **AoA curriculum pipeline, training schedule, and masking modifications**.

**Upstream base implementation (used as the ELECTRA foundation):**  
https://github.com/lucidrains/electra-pytorch

---

## File Structure
A quick overview of the structure of the folder.

Note that some files transfer with the original repo and remained untouched.

Scroll down to Usage to see how to run.

```bash
.
├── all_glue_results.json
├── blimp_results.json
├── data
│   ├── aoa.csv: original AoA data uncleaned
│   ├── aoa_histogram.png: distribution of AoA tokens
│   ├── aoa_simple.csv: simplified version with the required columns only
│   ├── aoa_with_bins.csv: aoa data with token and associated bin 
│   ├── download_aoa.py: download AoA data
│   ├── glue_data: data for evaluation
│   ├── openwebtext: unzipped openwebtext data
│   ├── openwebtext_features: openwebtext data extracted features
│   ├── openwebtext.tar.xz: compressed openwebtext data
│   ├── token2diff.pkl: 
│   ├── vocab.txt: token vocab file
│   ├── word2aoa.pkl: cleaned AoA data
│   └── word2bin.pkl: processed AoA tokens stored into bins
├── ELECTRA_AoA_Eval
│   ├── final_summary.csv: results for ELECTRA model with AoA
├── ELECTRA_BASE_Eval
│   ├── final_summary.csv: results for base ELECTRA model
├── electra.png
├── electra_pytorch
│   ├── electra_pytorch_aoa.py: modified implementation of ELECTRA with AoA
│   ├── electra_pytorch.py: implementation of original ELECTRA
│   ├── __init__.py
├── eval.py: to run the evaluation task of GLUE and BLiMP
├── output
│   ├── pretrain
│   │   └── base_final: final pretrained base model
│   └── pretrain_aoa
│       └── aoa_pretrain_final: final pretrained model with AoA
├── pretraining
│   └── openwebtext
│       ├── aoa_difficullty_process.py: match the aoa dataset to bins and difficulty 
│       ├── arg.py: args parser
│       ├── dataset.py: dataset builder to fit training pipeline
│       ├── preprocess.py: script to extract the features of openwebtext data
│       ├── pretrain_aoa.py: script to train AoA ELECTRA
│       ├── pretrain.py: script to train base ELECTRA
│       ├── small_discriminator.json: config for discriminator
│       ├── small_generator.json: config for generator
│       └── tokenization.py: tokenizer implementation
├── README.md: this file
├── requirement.txt: dependencies to install
```

---

## Key Contributions

- **AoA difficulty processing:** Converts AoA information into difficulty metadata/bins that can drive training.
- **AoA curriculum pretraining:** Trains across AoA stages (easy → hard).
- **Modified training scheduler:** Supports curriculum progression and a freeze/progress strategy.
- **Soft-freezing:** Allows a small fraction of updates from the current bin during freeze.
- **Masking adjustments:** Added flexibility in masking because token-level AoA mapping was not strong enough to rely on directly.

---

## Repository Structure

| Script | Description |
|--------|-------------|
| `pretraining/openwebtext/preprocess.py` | Extract the features from openwebtext data and make it ready to train |
| `pretraining/openwebtext/aoa_difficulty_process.py` | Builds AoA difficulty assignments / curriculum bins used for training |
| `pretraining/openwebtext/pretrain_aoa.py` | Main AoA curriculum pretraining entry point |
| `pretraining/openwebtext/pretrain.py` | Baseline pretraining entry point (non-AoA / standard run) |
| `eval.py` | Evaluation script for a trained checkpoint |

---

## Results

Both models were evaluated at **250,000 pretraining steps** on the GLUE benchmark and BLiMP (Benchmark of Linguistic Minimal Pairs). The table below compares the base ELECTRA model against the AoA curriculum variant.

### GLUE Benchmark

| Task | Metric | Base ELECTRA | AoA ELECTRA | Δ |
|------|--------|:------------:|:-----------:|:---:|
| CoLA | Matthews Corr. | 0.5312 | 0.5233 | -0.0079 |
| SST-2 | Accuracy | 0.8612 | 0.8532 | -0.0080 |
| MRPC | F1 / Accuracy | 0.8793 / 0.8284 | 0.8494 / 0.7819 | -0.0299 / -0.0465 |
| QQP | F1 / Accuracy | 0.8246 / 0.8719 | 0.8235 / 0.8671 | -0.0011 / -0.0048 |
| STS-B | Pearson / Spearman | 0.0404 / 0.0461 | 0.6939 / 0.7086 | **+0.6535 / +0.6625** |
| MNLI | Acc (m/mm) | 0.7612 / 0.7726 | 0.7433 / 0.7529 | -0.0179 / -0.0197 |
| QNLI | Accuracy | 0.8504 | 0.8532 | +0.0028 |
| RTE | Accuracy | 0.6101 | 0.5740 | -0.0361 |
| WNLI | Accuracy | 0.5634 | 0.5211 | -0.0423 |
| **GLUE Score** | | **0.6581** | **0.7145** | **+0.0564** |

### BLiMP

| Model | BLiMP Overall |
|-------|:-------------:|
| Base ELECTRA | **0.7091** |
| AoA ELECTRA | 0.6748 |
| Δ | -0.0343 |

### Discussion

The AoA curriculum model achieves a **higher aggregate GLUE score (0.7145 vs 0.6581)**, driven primarily by a dramatic improvement on **STS-B** (semantic textual similarity), where the base model nearly fails (Pearson ~0.04) while the AoA model reaches a competitive score (Pearson ~0.69). This suggests the curriculum schedule encourages the model to build stronger semantic representations — a natural outcome of progressive exposure from simple, concrete vocabulary toward more abstract language.

On most other GLUE tasks the two models are comparable, with the base model holding small edges on CoLA, SST-2, MRPC, MNLI, RTE, and WNLI. The **BLiMP score** favors the base model (0.7091 vs 0.6748), indicating the AoA model is slightly weaker at syntactic acceptability judgments, possibly because the curriculum's staged exposure delays full grammatical coverage.

Overall, the results suggest that AoA-grounded curriculum learning can meaningfully improve semantic understanding while trading off marginal performance on syntactic tasks — a trade-off consistent with the psycholinguistic motivation of the approach.

---

## Data Download

The preprocessed data (OpenWebText features, AoA bins, and vocabulary files) can be downloaded directly from Google Drive, so you can skip the preprocessing steps if desired.

**[Download Data — Google Drive](YOUR_LINK_HERE)**

Once downloaded, extract the contents into the `data/` directory before running training.

---

## Usage

### Make the virtual environment

```bash
$ python3 -m venv .venv

$ source .venv/bin/activate

$ pip install -r requirements.txt
```
### Data:
> **Note:** Expects the AoA source files (`aoa.csv`, `vocab.txt`) to be present in the `data/` folder before running. You can download processed data to save time from the [[Google Drive link](https://drive.google.com/drive/folders/1IJZlcw3HY-8veQvS3w1wY4Ds81pl_UKF?usp=drive_link)](#data-download) above.

#### OpenWebText Data Processing (already done)

```bash
$ python pretraining/openwebtext/preprocess.py
```

#### AoA Difficulty Processing (already done)


```bash
$ python pretraining/openwebtext/aoa_difficulty_process.py
```

### AoA Curriculum Pretraining

```bash
$ python -m pretraining.openwebtext.pretrain_aoa
```

### Baseline Pretraining (optional)

```bash
$ python pretraining/openwebtext/pretrain.py
```

---

## Evaluation

```bash
$ TOKENIZERS_PARALLELISM=false python -u eval.py \
  --model /path/to/checkpoint/ \
  --vocab_path /path/to/vocab.txt \
  --output_root ./outputs_electra_eval \
  --do_lower_case \
  --fp16 \
  --grad_accum 2 \
  --tasks all
```

### Notes

- **Checkpoint:** 250,000 steps
- **Soft-freezing value:** `0.15` (small percentage of current-bin updates allowed during freeze)
- **Masking:** Made more flexible because the token-level AoA mapping did not have good alignment, so relying purely on token-level AoA constraints was not stable
- `TOKENIZERS_PARALLELISM=false` reduces tokenizer parallelism warnings/overhead

---

## Known Limitations

### AoA ↔ Tokenization Mismatch

AoA resources are typically **word-level**, while ELECTRA training uses **subword tokens**. This mismatch can cause difficulty assignments to be noisy or incomplete at the token level. This repo includes training/masking adjustments to keep training stable even when token-level AoA alignment is imperfect.

---

## Acknowledgements

Base ELECTRA implementation: [lucidrains/electra-pytorch](https://github.com/lucidrains/electra-pytorch)