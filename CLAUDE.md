# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Course materials for **Redes Neuronales y SVM** at Universidad Anáhuac México (Prof. Dr. Aboud Barsekh-Onji). Contains lecture slides, MATLAB examples, and final project documentation.

## Identity
- Prof. Dr. Aboud Barsekh-Onji, Professor at the Faculty of Engineering, Universidad Anáhuac México
- Researche topics: hybrid models (Fuzzy Logic, PSO/MOPSO, LSTM), Fuzzy Substractive Clustering, Evolutionary computation, Many Objectives Optimization.

## Technical environment
- OS: Ubuntu 24.04, ThinkPad T14
- Python: conda env `research` (Python 3.11)
  - NEVER use venv/virtualenv, always conda
  - Interpreter: /home/aboudonji/miniforge3/envs/research/bin/python
- MATLAB: R2025b (main language)
- LaTeX: pdflatex by default, xelatex as fallback

## Delivery rules
- Academic documents: Markdown or LaTeX/Beamer (Berlin theme, 16:9)
- Presentations: Beamer, NOT PowerPoint
- Skills available in: ~/.config/claude/skills/

## Language
- Respond in the language in which the question is asked (ES/EN/AR)
## Compile LaTeX Slides (Beamer)

All presentations are in `Neural Network Conferences/`. Each `.tex` file is self-contained (no shared input preamble — `preamble.tex` is a standalone template, not `\input`'d).

```bash
# From within "Neural Network Conferences/"
pdflatex CNN.tex
pdflatex CNN.tex   # run twice for TOC/cross-refs
```

Or with latexmk (handles multiple passes automatically):
```bash
latexmk -pdf CNN.tex
```

Clean auxiliary files:
```bash
latexmk -c   # removes .aux, .log, .nav, .snm, .toc, .out, .fls, .fdb_latexmk
```

## Run MATLAB Examples

MATLAB R2025b is the primary environment. Scripts are in `Examples/ExN/` and require the Deep Learning Toolbox.

```matlab
% From MATLAB Command Window (cd to the example folder first)
run('Examples/Ex1/NN_ex1.m')
```

Scripts follow a numbered workflow: data loading → preprocessing → training options → `trainnet()` → evaluation with `minibatchpredict` + `scores2label`.

## Architecture

```
Neural Network Conferences/   ← Beamer slides (.tex → .pdf), Berlin theme, aspectratio=169
  Figures/                    ← Shared figures used across slides
Examples/
  Ex1/                        ← LSTM for time-series classification (WaveformData)
  Ex2/                        ← Generated data, custom NN
  Ex3/                        ← CNN with Deep Network Designer
  Ex4/                        ← CNN activity (Markdown instructions)
  Ex5_KnowledgeTransfer/      ← Transfer learning example
  SVM/                        ← SVM introduction + MATLAB script
  optimizers/                 ← Manual backprop, SGD, ADAM worked examples
Final Projects/               ← LaTeX book-class documents (rubric, project guidelines)
Books_Articles/               ← Reference PDFs
```

## LaTeX Conventions

- **Slides**: `\documentclass[aspectratio=169,xcolor={...}]{beamer}` with `\usetheme{Berlin}`
- **Documents**: `\documentclass[12pt,letterpaper]{book}` with Spanish babel (`es-tabla`)
- Standard math packages: `amsmath`, `amsfonts`, `amssymb`; code listings via `listings` with `MATLABStyle` (currently commented out in preamble — uncomment when adding code blocks)
- Compile with `pdflatex`; use `xelatex` only if font issues arise

## MATLAB Code Style

- Comments in Spanish; section headers with `%%`
- Use `trainnet()` (R2023a+ API), not the legacy `train()`
- Training options via `trainingOptions("adam", ...)` with named arguments
- Standard evaluation pipeline: `minibatchpredict` → `scores2label` → `confusionchart`
