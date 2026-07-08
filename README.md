# earthquake-transformer-kd
# Compressing the Earthquake Transformer via Knowledge Distillation

**Final Project – Phase A (61998) | Braude College of Engineering**
Software Engineering Department

Project code: 26-2-R-3

## Authors
- Or Shterenshus
- Shiraz Balmas Shterenshus

**Advisors:** Dr. Elena Kramer, Dr. Dan Lemberg

## Overview
This project investigates compressing the **Earthquake Transformer (EQTransformer)**
using **Knowledge Distillation**, in order to reduce model size, memory usage,
and inference time while preserving earthquake detection and P/S-wave phase-picking
performance. The goal is to enable efficient real-time seismic analysis on
resource-constrained edge devices.

Inspired by DistilBERT, a compressed **student** model is trained to reproduce the
probability curves of the original EQTransformer **teacher** model.

## Datasets
- **STEAD** – Stanford Earthquake Dataset (training & evaluation)
- **K-NET** – Japanese strong-motion dataset (cross-domain evaluation)

## Repository Structure
| Path | Description |
|------|-------------|
| `docs/` | Project book and presentation |
| `notebooks/` | Experiments (baseline EQTransformer validation) |
| `results/` | Experiment outputs and figures |
| `requirements.txt` | Python dependencies |

## Baseline Experiment (Phase A)
A validation run of the pretrained EQTransformer on 100 STEAD sample recordings:
- Detected **99/100** seismic events
- Model size: **373,495 parameters (~1.4 MB)**
- Establishes the reference baseline for compression experiments

## Tools & Technologies
Python · TensorFlow / Keras · EQTransformer · Google Colab · Git

## Status
🚧 Phase A (research proposal). Compression and distillation experiments
will be implemented in Phase B.
