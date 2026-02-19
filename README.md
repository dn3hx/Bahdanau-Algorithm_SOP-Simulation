# Bahdanau-Algorithm_SOP-Simulation
Simulation Code for the original vs enhanced Bahdanau Algorithm
# Enhanced Bahdanau Attention via Social Metadata Injection

## Overview

This repository contains a simulation demonstrating the algorithmic enhancement of the Bahdanau Additive Attention Scoring Function through Social Metadata Injection.

The purpose of this project is to illustrate the mathematical and computational difference between:

1. The original Bahdanau Attention algorithm
2. The enhanced Bahdanau Attention algorithm incorporating social metadata

This simulation supports the Statement of the Problem (SOP 1) of the study:

> Limited Contextual Awareness of the Standard Bahdanau Attention Scoring Algorithm

---

## Background

The original Bahdanau Attention scoring function computes alignment scores using only textual hidden representations:

e_ij = v^T tanh(W_s s_t + W_h h_i)

However, real-world conversational toxicity is influenced not only by linguistic content but also by:

- User roles
- Prior warnings
- Interaction history
- Reply structure
- Temporal behavior

To address this limitation, the enhanced algorithm modifies the scoring function to:

e_ij = v^T tanh(W_s s_t + W_h h_i + W_m m_i)

Where:
- m_i = social metadata vector
- W_m = learnable metadata weight matrix

This modification expands the alignment space from purely linguistic features to a joint linguistic-social feature space.

---

## What This Simulation Demonstrates

This project simulates:

- Encoder hidden states
- Decoder hidden state
- Social metadata vector

It computes attention weights using:

- Original Bahdanau Attention
- Enhanced Bahdanau Attention (with metadata injection)

The simulation then visualizes:

1. Side-by-side comparison of attention weights
2. Attention shift caused by metadata injection

The difference in attention distribution proves that metadata injection modifies the alignment scoring behavior.

---

## Requirements

This project runs in:

- Google Colab (recommended)
- Python 3.x
- PyTorch

If running locally:

```bash
pip install torch matplotlib numpy

