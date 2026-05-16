# Multi-Modal Chest X-ray Intelligence System

## Overview

This project implements a dual-mode multi-modal medical AI system for chest X-ray analysis using MedGemma and ColPali.

The system supports:

1. Chest X-ray Report Generation
2. RAG-based Medical Question Answering

---

# Models Used

- MedGemma → report generation and answer generation
- ColPali → retrieval for the RAG pipeline

---

# Features

- Multi-modal image + text processing
- Structured radiology report generation
- Retrieval-Augmented Generation (RAG)
- Medical question answering
- Gradio demo interface
- Evaluation metrics and model comparison

---

# Architecture

## Mode 1: Report Generation

```text
Chest X-ray Image
        ↓
     MedGemma
        ↓
Findings + Impression
```

## Mode 2: RAG Medical QA

```text
Medical Question
        ↓
   ColPali Retrieval
        ↓
Retrieved Context
        ↓
     MedGemma
        ↓
Grounded Answer
```

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run Demo

```bash
python app.py
```

---

# Example Questions

```text
What are the findings in this chest X-ray?
```

```text
Is there pleural effusion?
```

```text
Does the patient have pneumonia?
```

---

# Evaluation

- ROUGE scores for report generation
- QA accuracy
- Retrieval semantic evaluation



