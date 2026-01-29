# Math Misconception Detection using NLP (MAP Competition)

This repository contains an end-to-end Natural Language Processing (NLP) solution for detecting and classifying **student math misconceptions** from open-ended explanations.  
The project was developed as part of the **Misconception Annotation Project (MAP)** Kaggle competition.

The goal is to automatically diagnose *how* a student is thinking incorrectly — not just whether an answer is right or wrong — enabling scalable, actionable feedback for educators.

---

## Problem Statement

Students often provide written explanations when solving math problems. These explanations reveal valuable insights into their reasoning, including:

- Correct conceptual understanding  
- Incomplete or irrelevant reasoning  
- Systematic misconceptions (e.g., whole-number bias in decimals)

Manually analyzing such explanations is time-consuming and difficult to scale.

### Objective
Build an NLP-based machine learning model that:

1. Determines whether a student’s selected answer is **correct or incorrect**
2. Identifies whether the explanation shows a **misconception**
3. Predicts the **specific type of misconception**, if present
4. Ranks the **top 3 most likely diagnoses** per response

---

## Key Concept: Category : Misconception

Each prediction is a combined label of the form: Category: Misconception


Examples:
- `True_Correct:NA`
- `False_Misconception:Whole_numbers_larger`
- `True_Misconception:Incomplete`

The model is evaluated using **MAP@3 (Mean Average Precision @ 3)**, which rewards correct ranking of the true diagnosis within the top three predictions.

---

## Dataset Overview

The dataset is provided by **Eedi**, an educational platform focused on diagnosing math misconceptions.

Each row contains:
- `QuestionText` – the math problem
- `MC_Answer` – the selected multiple-choice answer
- `StudentExplanation` – open-ended reasoning
- `Category` (train only)
- `Misconception` (train only)

The test set contains ~16,000 student responses.

---

## Approach & Architecture

### 1. Text Construction
We combine multiple information sources into a single input sequence: 
[QUESTION] Question text
[ANSWER_SELECTED] Student's chosen option
[EXPLANATION] Student explanation


This provides full contextual grounding for the model.

---

### 2. Label Engineering
- `Category` and `Misconception` are merged into a single label
- Example: `False_Misconception:Whole_numbers_larger`
- Each unique label is mapped to an integer ID

---

### 3. Model
- Transformer-based sequence classifier
- **DeBERTa v3 (small)** used for stability and efficiency on Kaggle
- Classification head predicts logits over all label classes

---

### 4. Training Strategy
- Hugging Face `Trainer`
- Step-limited training for stability (`max_steps`)
- Infrequent evaluation to avoid dataloader deadlocks
- Only **one checkpoint** saved at any time (disk-safe)
- Automatic best-model selection using MAP@3

---

### 5. Evaluation Metric
**MAP@3** rewards correct ranking of the true label:
- Rank 1 → score 1.0
- Rank 2 → score 0.5
- Rank 3 → score 0.33
- Not in top 3 → score 0

---

##  Features Implemented

###  Core Features
- End-to-end data preprocessing pipeline
- Robust label mapping and encoding
- Transformer-based text classification
- Custom MAP@3 evaluation metric
- Top-3 prediction generation
- Kaggle-safe checkpointing and recovery
- Disk- and time-efficient training configuration

###  Engineering Features
- Absolute-path checkpointing
- Resume-safe training
- Inference-only recovery after kernel reset
- Prediction analysis dataframe with logits and probabilities

---

##  Output Artifacts

The project generates:
- `submission.csv` – Kaggle-ready submission file
- Analysis dataframe with:
  - Top-3 predicted labels
  - Prediction confidence scores
  - Raw logits (optional)

---

##  Results

- Training completed successfully within ~2–3 hours
- Achieved competitive **MAP@3 scores (~0.75–0.80)** on validation
- Stable training without disk overflow or kernel crashes

---

##  Technologies Used

- Python 3
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy / Pandas
- Scikit-learn
- Kaggle GPU environment

---







