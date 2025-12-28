# Federated Learning for Financial Sentiment Analysis using FinBERT

This project applies **Federated Learning** to financial sentiment analysis using **FinBERT**.  
We compare three algorithms:

✔ FedAvg  
✔ FedProx  
✔ Adaptive Aggregation (Adaptive FedAvg)  

Our dataset contains financial text from **Twitter, News, and Financial Reports**, split across clients to simulate privacy-preserving training.

---

## 🧠 Objective
To study how different federated optimization algorithms perform on financial sentiment tasks, and whether FedProx or Adaptive aggregation improve stability and accuracy over FedAvg.

---

## 📂 Repository Structure

```
Project/
├── 01_data_normalisation.ipynb
├── 02_federated_setup.ipynb
├── 03_local_finbert_twitter.ipynb
├── 04_local_finbert_news.ipynb
├── 05_local_finbert_reports.ipynb
├── 06_federated_fedavg_finbert.ipynb
├── 07_federated_fedavg_multiround.ipynb
├── 08_federated_fedprox_multiround.ipynb
├── 09_federated_adaptive_aggregation.ipynb
├── 10_federated_results_comparison.ipynb
├── data.zip   ← dataset raw,clean,processed,val_splits
├── results.zip ← output metrics & plots

```

---

## 📊 Key Findings

| Method | Final Avg F1 |
|-------|--------------|
| **FedAvg (10 rounds)** | ~0.846 |
| **FedProx (μ = 0.05)** | ~0.855 |
| **Adaptive FedAvg**   | ~0.823 |

FedProx showed **slightly higher and more stable performance**, especially where client data distributions differed.

---

## 🏗 Model Weights
Model folders were too large for GitHub.  
They can be shared via cloud storage if needed.

---

## 🚀 Tech Stack
- Python / PyTorch
- HuggingFace Transformers
- Sklearn
- Federated Learning concepts

---

## 📌 Notes
This is a student research project intended for learning & academic exploration.
