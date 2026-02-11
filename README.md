# Federated Learning for Financial Sentiment Analysis using FinBERT

This project applies **Federated Learning (FL)** to **financial sentiment analysis**
using the pretrained language model **FinBERT**.  
The goal is to evaluate how different FL optimization strategies perform when training
across multiple private financial text sources without sharing raw data.

We compare three algorithms:

✔ **FedAvg** (baseline)  
✔ **FedProx** (proximal stability term)  
✔ **Adaptive Aggregation** (performance-weighted averaging)  

Training data is split across **three simulated clients**:

• Financial Twitter posts  
• Financial News headlines  
• Financial Reports / Statements  

to mimic privacy-preserving, real-world deployment.

---

## 🎯 Objective

To study whether **FedProx or Adaptive Aggregation** improve stability,
neutral-class handling, and overall sentiment accuracy  
compared to standard **FedAvg**, when training on **non-IID financial datasets**.

---

## 🚀 Try the Models — Google Colab Demo

You can test the trained models directly in Colab  
without downloading anything:

👉 **Federated FinBERT Demo Notebook**  
(loads FedAvg / FedProx / Adaptive and runs predictions)

> `Federated_FinBERT_Demo.ipynb`

Features:
✔ model-selection menu  
✔ simple `predict_sentiment("text")` function  
✔ prints sentiment + class probabilities  

---

## 🤗 HuggingFace Model Links

The trained models are publicly hosted:

🔹 **FedAvg**  
https://huggingface.co/harshprasad03/FinBERT-FedAvg  

🔹 **FedProx**  
https://huggingface.co/harshprasad03/FinBERT-FedProx  

🔹 **Adaptive Aggregation**  
https://huggingface.co/harshprasad03/FinBERT-AdaptiveFedAvg  


Each model card contains:

• description  
• training setup  
• usage example  
• authorship credit  

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
├── Federated_FinBERT_Demo.ipynb
├── data.zip
├── results.zip

```

---

## 📊 Key Experimental Findings

| Method | Final Avg F1-Score |
|-------|--------------------|
| **FedAvg (10 rounds)** | ~0.835 |
| **FedProx (μ = 0.05)** | **~0.855** |
| **Adaptive FedAvg** | ~0.823 |

### 🧠 Interpretation
✔ **FedProx produced the most stable and balanced performance**,  
particularly for Neutral sentiment classification

✔ **FedAvg showed mild optimistic bias**  
(tending to classify factual statements as Positive)

✔ **Adaptive Aggregation converged smoothly but did not outperform FedProx or basic FedAvg**

These results support the idea that **FedProx reduces client drift**
in non-IID federated financial text settings.

---

## 🏗 Model Architecture

Base model:
```

ProsusAI/finbert

```

Task:
```

3-class sentiment
(Positive / Neutral / Negative)

```

Federated setup:
```

3 clients
10 global rounds
3 local epochs

```

---

## 🧪 Datasets

Three private client datasets:

• Financial Twitter posts  
• Financial News  
• Financial Reports  

Raw/processed/split data stored in:

```

data.zip

```

Validation splits used for global evaluation(present in data folder under splits).

---

## 🛠 Tech Stack

• Python  
• PyTorch  
• HuggingFace Transformers  
• Sklearn  
• Jupyter / Google Colab  

---

## 📌 Project Status

This work forms part of a **Master’s-Level Research Project**
on **Federated NLP Applications in Finance**.

The project demonstrates:

✔ privacy-preserving training  
✔ domain-specific language modeling  
✔ algorithm comparison  
✔ reproducible experiments  

---

## 👥 Authors

**Harsh Prasad**  
**Sai Dhole**  

---

## 📜 License

This repository is released under the **MIT License**.

---

## 🙏 Acknowledgements

This project is built on:

`ProsusAI/finbert`  
and the HuggingFace ecosystem.

---
