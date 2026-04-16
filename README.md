# 🛒 Retail Customer Intelligence Engine

> An end-to-end AI system that predicts customer churn, segments customers by behavior, estimates lifetime value, and classifies customer feedback sentiment — combining structured behavioral data with unstructured text into a unified customer intelligence platform.

**🔗 [Live Demo](https://retail-cutomer-engine.streamlit.app/)** | **[GitHub Repository](https://github.com/Fawazahmad75/retail-customer-intelligence-engine)**

---

## 📖 Project Overview

Working at Walmart, I see firsthand how customers leave for predictable reasons — a bad pickup experience, a slow delivery, an unresolved complaint. This project automates the pattern recognition that experienced retail workers do intuitively, combining behavioral analytics with feedback analysis to give businesses a complete picture of every customer.

The system answers four critical business questions about any customer:
1. **Will this customer churn?** (Predictive ML)
2. **Why are they at risk?** (SHAP explainability)
3. **Are they worth retaining?** (RFM segmentation + CLV)
4. **What are they actually saying?** (NLP sentiment analysis)

---

## 🎯 Problem Statement

Subscription and e-commerce businesses lose 15-30% of customers annually. Most retention systems fail because they:
- Treat all churn as equally bad (it's not — losing a $12 customer doesn't justify a $20 retention offer)
- Rely on structured data alone, missing signals in customer feedback
- Provide predictions without explanations, leaving teams unable to act
- Don't differentiate between high-value loyalists and low-value drifters

This project addresses all four gaps with a unified, business-aware AI system.

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Languages** | Python 3.12 |
| **ML/DL** | scikit-learn, PyTorch, XGBoost (Gradient Boosting fallback) |
| **NLP** | TF-IDF, scikit-learn |
| **Explainability** | SHAP (TreeExplainer) |
| **Data** | pandas, NumPy |
| **Visualization** | matplotlib, seaborn |
| **Deployment** | Streamlit, Streamlit Cloud |
| **Version Control** | Git, GitHub |

---

## 📊 Project Components

### 1. Customer Churn Prediction

Trained and compared **four models** on a 5,630-customer e-commerce dataset to predict churn:

| Model | Train Acc | Test Acc | F1 (Churn) | Churner Recall | AUC | Overfitting Gap |
|-------|-----------|----------|------------|----------------|-----|-----------------|
| Logistic Regression | 89.9% | 88.8% | 0.611 | 52% | 0.886 | 1.1% |
| Gradient Boosting | 93.3% | 91.8% | 0.728 | 65% | 0.943 | 1.5% |
| **Random Forest (default)** | **100%** | **98.3%** | **0.948** | **91%** | **0.999** | **1.7%** |
| Random Forest (tuned, depth=10) | 97.0% | 94.0% | 0.793 | 67% | 0.982 | 3.0% |
| PyTorch Neural Network | 97.8% | 95.6% | 0.858 | 79% | 0.989 | 2.2% |

**📈 Insert plot here:** `plots/roc_curves.png` — *ROC curve comparison across all models*

**📈 Insert plot here:** `plots/model_comparison.png` — *Performance comparison table*

#### Key Decisions:

- **Why Random Forest won over PyTorch:** On small tabular datasets with strong feature signals (tenure + complaint), tree ensembles outperform deep learning because they natively handle mixed feature types and capture non-linear thresholds without requiring scaling. The PyTorch NN was included to validate this choice empirically rather than assume it.

- **Why I chose default RF over the tuned version:** The tuned model had a smaller overfitting gap but dropped churner recall from 91% to 67%. In a business context, missing churners is more costly than the slight overfitting risk — confirmed by the strong test set performance (98.3%).

- **Why F1 and recall matter more than accuracy:** Only 16.8% of customers churned. A naive "predict no churn" model would achieve 83% accuracy while catching zero churners. F1 and recall measure what actually matters — finding the customers about to leave.

#### Confusion Matrix (Random Forest):
- **935** correctly predicted to stay (true negatives)
- **172** correctly predicted to churn (true positives)
- **1** false alarm (false positive)
- **18** missed churners (false negatives)

**📈 Insert plot here:** `plots/confusion_matrix.png`

---

### 2. Custom PyTorch Neural Network

Built from scratch to demonstrate deep learning fundamentals:

```
ChurnNet:
  Linear(29 → 64) + ReLU + Dropout(0.3)
  Linear(64 → 32) + ReLU + Dropout(0.3)
  Linear(32 → 1)  + Sigmoid
```

- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (lr=0.001)
- **Training:** 100 epochs, batch size 32, with feature scaling via StandardScaler
- **Custom Dataset class** wrapping pandas DataFrames into PyTorch tensors
- **Manual training loop** with `optimizer.zero_grad()`, forward pass, loss computation, backward pass, and weight updates
- **Final test F1:** 0.858 (second-best across all models)

The training loss decreased smoothly from 0.24 to 0.097 across 100 epochs, indicating healthy learning without instability.

---

### 3. SHAP Explainability

Implemented SHAP (SHapley Additive exPlanations) to interpret predictions, transforming the black-box Random Forest into an explainable system.

**Top 5 churn drivers identified:**
1. **Tenure** (low tenure → high churn risk)
2. **Complain** (filed complaint → high churn risk)
3. **DaySinceLastOrder** (long gaps → high churn risk)
4. **MaritalStatus_Single** (single customers churn more)
5. **NumberOfAddress** (more saved addresses → higher churn — possible indicator of unstable shopping habits)

**📈 Insert plot here:** `plots/shap_summary.png` — *SHAP beeswarm plot showing feature impact direction and magnitude*

**📈 Insert plot here:** `plots/feature_importance.png` — *Random Forest feature importance for comparison*

I included both Random Forest's built-in feature importance and SHAP values because they measure different things: built-in importance tracks usage frequency in tree splits, while SHAP measures actual contribution to individual predictions using game theory. SHAP is more reliable for explaining specific customer predictions.

---

### 4. RFM Customer Segmentation

Implemented **Recency-Frequency-Monetary (RFM) analysis** to segment customers into actionable groups:

| Segment | Churn Rate | Strategy |
|---------|-----------|----------|
| **Needs Attention** | 26.9% | Top priority — targeted promotions, satisfaction surveys |
| **New Customers** | 21.6% | Onboarding campaigns, first-purchase discounts |
| **At Risk** | 18.4% | Personalized retention offers, free shipping |
| **Champions** | 15.5% | Reward with loyalty programs, VIP treatment |
| **Loyal** | 11.5% | Maintain engagement, ask for referrals |
| **Lost** | 9.6% | Minimal investment — let go |

**📈 Insert plot here:** `plots/churn_vs_clv.png` — *Customer segments plotted by churn risk vs lifetime value*

**Key insight:** Counterintuitively, "Lost" customers have the lowest current churn rate in our active dataset — likely because truly lost customers have already left and the remaining low-activity customers are stable infrequent buyers. The largest opportunity is the "New Customers" segment: large group, high churn, low CLV. Improving onboarding moves them into "Loyal," which is where the real value lives.

---

### 5. Customer Lifetime Value (CLV) Estimation

Implemented a simple but effective CLV formula: `CashbackAmount × OrderCount × (Tenure / 12)`

This converts raw predictions into business-meaningful dollar amounts. The system can now answer: "Is this $20 retention offer worth sending?" by comparing it against the customer's estimated CLV.

**Key business insight:** Not all churn is worth preventing. A customer in the "Lost" segment with CLV of $12 isn't worth a $20 retention offer. Focus retention budgets on "At Risk" and "Champions" segments where the math works.

---

### 6. NLP Sentiment Classifier

Built a text classifier on **22,641 customer reviews** (Women's Clothing E-Commerce dataset):

| Model | Accuracy | F1 (weighted) | **Negative Recall** | Negative Precision | AUC |
|-------|----------|---------------|---------------------|--------------------|----|
| **Logistic Regression** ✅ | 86% | 0.913 | **82%** | 59% | 0.926 |
| Random Forest | 86% | 0.920 | 33% | 77% | 0.906 |

**📈 Insert plot here:** `plots/nlp_model_comparison.png` — *Side-by-side comparison highlighting the recall gap*

#### Why I chose Logistic Regression despite Random Forest's higher F1:

Random Forest achieved a slightly higher weighted F1, but **Logistic Regression dramatically outperformed it on the minority class** (negative reviews): **82% recall vs 33%**. In a business context, Random Forest would miss two-thirds of unhappy customers — exactly the opposite of what you want from a sentiment analysis system.

This taught me a critical lesson: **headline metrics can hide critical business failures**. Choosing the right metric matters more than chasing the highest number. I deployed Logistic Regression because catching unhappy customers is the entire point of the system.

#### Production-Aware Confidence Threshold:

I added a 65% confidence threshold to the deployed classifier. When the model is uncertain (confidence < 65%), it flags the review as "Uncertain — Needs Human Review" rather than forcing a potentially wrong classification. This is how real production sentiment systems handle edge cases — admitting uncertainty is more valuable than pretending confidence.

---

## 🎨 Live Application Features

The deployed Streamlit app combines all components into four tabs:

### Tab 1: Churn Predictor
- 13 input fields for customer behavior data
- Real-time churn probability + risk level (Low/Medium/High)
- **Integrated RFM segment** based on input
- **Estimated Lifetime Value** in dollars
- Context-aware retention recommendations per segment

### Tab 2: Review Classifier
- Free-text input for customer reviews
- Sentiment classification (Positive / Negative / Uncertain)
- Confidence score
- Production threshold for human review routing

### Tab 3: Full Analysis (The Killer Feature)
- Combines behavioral data AND review text in one input
- Generates a unified intelligence report including:
  - Churn risk
  - RFM segment
  - Lifetime value
  - Review sentiment
- Context-aware combined insight that handles four scenarios:
  1. **High risk + Negative feedback** → Critical intervention needed
  2. **High risk + Positive feedback** → Quiet disengagement, proactive outreach
  3. **Low risk + Negative feedback** → Vocal complaint but stable, quick fix needed
  4. **Low risk + Positive feedback** → Healthy customer, maintain engagement

### Tab 4: About the Project
- Technical overview, methodology, and project context

---

## 📁 Project Structure

```
retail-customer-intelligence-engine/
├── app.py                          # Streamlit application (deployed)
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── models/
│   ├── churn_rf.joblib            # Random Forest (deployment model)
│   ├── churn_lr.joblib            # Logistic Regression
│   ├── churn_gbc.joblib           # Gradient Boosting
│   ├── churn_nn.pth               # PyTorch Neural Network
│   ├── scaler.joblib              # StandardScaler for PyTorch input
│   ├── text_classifier.joblib     # NLP Logistic Regression
│   └── tfidf_vectorizer.joblib    # TF-IDF vectorizer
├── plots/
│   ├── roc_curves.png             # Model comparison ROC curves
│   ├── confusion_matrix.png       # Random Forest confusion matrix
│   ├── shap_summary.png           # SHAP explainability beeswarm
│   ├── feature_importance.png     # Random Forest feature importance
│   ├── churn_vs_clv.png          # Customer segments visualization
│   ├── nlp_model_comparison.png   # NLP classifier comparison
│   └── model_comparison.png       # Comprehensive model table
└── notebooks/
    ├── 01_data_cleaning_and_models.ipynb
    ├── 02_pytorch_model.ipynb
    └── 03_nlp_classifier.ipynb
```

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/Fawazahmad75/retail-customer-intelligence-engine.git
cd retail-customer-intelligence-engine

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📚 Datasets

1. **E-commerce Customer Churn** — 5,630 customers with 20 features
   [Kaggle Source](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

2. **Women's Clothing E-Commerce Reviews** — 22,641 customer reviews with ratings and recommendations
   [Kaggle Source](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

---

## 💡 Key Engineering Lessons

This project taught me several lessons that go beyond model accuracy:

1. **Feature engineering can hurt as much as help.** When I added RFM and CLV as model features, performance dropped because they were derived from features already in the model. I learned to use them as outputs (business intelligence) rather than inputs (model features).

2. **Headline metrics hide critical failures.** Random Forest's slightly higher NLP F1 masked the fact that it missed 67% of negative reviews. Always inspect per-class performance, especially on imbalanced data.

3. **Production systems need confidence thresholds.** Forcing predictions on uncertain inputs creates wrong answers. Routing uncertain cases to humans is engineering maturity.

4. **Tree-based models still win on tabular data.** Despite all the deep learning hype, my custom PyTorch network couldn't beat Random Forest on structured customer data. Use the right tool for the job.

5. **Business context beats technical complexity.** A simple Logistic Regression with the right metric beats a complex model optimizing the wrong thing.

---

## 🚧 Future Improvements

- **Real-time monitoring** for data drift and model performance degradation
- **Cloud deployment** to AWS/GCP with Docker containerization
- **A/B testing framework** to measure retention campaign effectiveness
- **FastAPI backend** for separation of inference logic from UI
- **Enhanced NLP** with fine-tuned transformer models (BERT/DistilBERT) for better subtle sentiment detection
- **Hybrid retrieval** for the planned RAG document assistant (Project 2)
- **Automated alerting** when "At Risk" segment grows beyond threshold

---

## 👤 About Me

Built by **Fawaz Ahmad** as part of my application for the **Junior Applied AI Engineer** position at **Genesis AI Garage** (St. John's, NL).

Currently working at Walmart, where the patterns I see daily inspired this project. I'm passionate about applying AI to real business problems where the stakes are tangible and the impact is measurable.

- **GitHub:** [@Fawazahmad75]
- **Live Demo:** (https://github.com/Fawazahmad75)

---
