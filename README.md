# 📚 Top-N Recommendation System with Implicit Feedback

> Comparative Analysis of Collaborative Filtering Algorithms: IBCF, ALS, and BPR for Top-10 Book Recommendations

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/Academic-License-green.svg)](#)

---

## 📌 Project Overview

This project focuses on developing a **Top-10 Recommendation System** using only **implicit feedback** data, without explicit ratings or review scores. Instead of relying on user-provided ratings, the system learns user preferences from interaction patterns such as clicks, views, or purchases to predict the ten most relevant items for each user.

### Problem Description

The dataset consists solely of `user_id` and `item_id` pairs, where each record indicates a positive interaction between a user and an item. Since no explicit ratings are available, the recommendation system must infer user preferences directly from historical interaction data.

**Main Challenges:**

* **High sparsity** — most users interact with only a small fraction of the available items
* **No explicit negative feedback** — lack of interaction does not necessarily imply dislike
* **Noisy data** — interactions do not always represent true user preferences

### Objective

To generate a ranked **Top-10 recommendation list** for each user based on historical interactions and evaluate recommendation quality using **MAP@10 (Mean Average Precision at 10)**.

---

## 📊 Dataset

| Property           | Value                                         |
| ------------------ | --------------------------------------------- |
| Source             | `train.csv` (data-books dataset)              |
| Total Interactions | 269,764                                       |
| Unique Users       | 13,876                                        |
| Unique Items       | 103,454                                       |
| Data Type          | Implicit Feedback (`user_id`, `item_id`)      |
| Data Split         | ~80% Training / ~20% Test (per-user hold-out) |

The dataset is highly sparse because each user interacts with only a small subset of the entire item catalog.

---

## 🧠 Implemented Models

### 1. Item-Based Collaborative Filtering (IBCF) — *Baseline Model*

**File:** `IBCF.ipynb`

A memory-based recommendation approach that computes item similarities based on **co-occurrence patterns** in user interactions.

**Main Steps:**

* Apply IDF-style weighting: `weight = 1 / log(1 + item_frequency)` to reduce popularity bias
* Transpose the user–item matrix into an Item×User representation
* Compute cosine similarity using `NearestNeighbors` (K=100 neighbors)
* Generate Top-10 recommendations by aggregating weighted similarity scores

**Results:**

| Stage    | MAP@10 |
| -------- | ------ |
| Baseline | 0.0136 |

---

### 2. Alternating Least Squares (ALS) — *Advanced Model*

**File:** `ALS_FINAL.ipynb`

A model-based recommendation approach using **Matrix Factorization** with confidence weighting, optimized for implicit feedback through the `implicit` library.

**Main Steps:**

* Convert the interaction matrix into a confidence matrix: `C(u,i) = 1 + α × r(u,i)`
* Factorize the matrix into low-dimensional latent user and item factors
* Alternately optimize user and item latent vectors until convergence
* Predict recommendation scores using latent vector dot products

**Results:**

| Stage    | MAP@10 |
| -------- | ------ |
| Baseline | 0.0459 |
| Tuned    | 0.0477 |

---

### 3. Bayesian Personalized Ranking (BPR) — *Advanced Model*

**File:** `BPR_Final.ipynb`

A pairwise ranking approach specifically designed for implicit feedback recommendation tasks. BPR learns to rank interacted items above non-interacted items using a log-sigmoid ranking loss.

**Main Steps:**

* Sample training triplets `(user, positive_item, negative_item)`
* Optimize pairwise BPR loss with regularization
* Utilize Adam optimization and negative sampling
* Tune learning rate, regularization strength, latent factors, and negative sample size

**Results:**

| Stage    | MAP@10     |
| -------- | ---------- |
| Baseline | 0.0493     |
| Tuned    | **0.8339** |

---

## 📈 Performance Comparison

| Model    | Approach Type                      | MAP@10 (Baseline) | MAP@10 (Tuned) | Description                                               |
| -------- | ---------------------------------- | :---------------: | :------------: | --------------------------------------------------------- |
| **IBCF** | Memory-Based                       |       0.0136      |        —       | Lowest performance, serving as the baseline               |
| **ALS**  | Model-Based (Matrix Factorization) |       0.0459      |     0.0477     | Improved performance compared to IBCF                     |
| **BPR**  | Model-Based (Pairwise Ranking)     |       0.0493      |   **0.8339**   | Highest performance with the most significant improvement |

> **BPR achieved the highest MAP@10 score**, demonstrating that ranking-based optimization is substantially more effective than item-similarity and matrix-reconstruction approaches for highly sparse implicit-feedback recommendation tasks.

---

## 🔍 Evaluation Metric: MAP@10

**Mean Average Precision at 10 (MAP@10)** evaluates both recommendation relevance and ranking quality within the Top-10 recommendation list.

$$\text{MAP@10} = \frac{1}{|U|} \sum_{u \in U} \text{AP@10}(u)$$

* Higher scores are awarded when relevant items appear near the top of the recommendation list
* Evaluates ranking quality rather than merely identifying relevant items
* Produces a single metric summarizing recommendation performance across all users

---

## 🛠️ Technologies Used

| Tool                                     | Purpose                                  |
| ---------------------------------------- | ---------------------------------------- |
| Python 3                                 | Main programming language                |
| Jupyter Notebook / Google Colab / Kaggle | Development environment                  |
| Pandas & NumPy                           | Data manipulation and processing         |
| SciPy (CSR Matrix)                       | Sparse matrix handling                   |
| scikit-learn                             | NearestNeighbors implementation for IBCF |
| `implicit` Library                       | ALS implementation                       |
| PyTorch                                  | BPR implementation                       |
| GPU (CUDA)                               | Model training acceleration              |

---

## 📁 Project Structure

```text
📦 top-n-recommendation-implicit-feedback
├── 📓 ALS FINAL.ipynb
├── 📓 BPR.ipynb
├── 📓 IBCF.ipynb
├── 📄 README.md
└── 📄 train.csv
```
---

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Jufourlisasirait/Top-N-Recommendation-Implicit-Feedback---09.git
cd Top-N-Recommendation-Implicit-Feedback---09
```

### 2. Install Dependencies

```bash
pip install pandas numpy scipy scikit-learn implicit torch
```

### 3. Prepare the Dataset

Place `train.csv` containing the following columns:

```text
user_id,item_id
```

in the project root directory.

### 4. Run the Notebooks

Execute the notebooks in the following order:

1. `IBCF.ipynb`
2. `ALS_FINAL.ipynb`
3. `BPR_Final.ipynb`

### 5. Use GPU (Optional)

GPU acceleration is recommended, particularly for training the BPR model using Google Colab or Kaggle.

---

## 🔑 Key Findings

* **IBCF** achieved the lowest performance and was highly affected by data sparsity.
* **ALS** improved recommendation quality by capturing latent user-item relationships through matrix factorization.
* **BPR delivered the best overall performance**, achieving the highest MAP@10 score after hyperparameter tuning.
* **Pairwise ranking optimization** proved more suitable for implicit feedback scenarios because it directly optimizes recommendation ranking.
* **Hyperparameter tuning played a critical role**, especially for BPR, which exhibited the largest performance improvement among all evaluated models.

---

*Final Project — Recommendation Systems Course*
*Institut Teknologi Del — 2025*
