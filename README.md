<div align="center">

# CORAL

> Stopping spam and malicious links before they ever reach a user, with a machine learning microservice delivering real-time message classification at production speed.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-production--ready-009688?style=flat-square&logo=fastapi&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-model-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pydantic](https://img.shields.io/badge/pydantic-validated-E92063?style=flat-square&logo=pydantic&logoColor=white)

![Accuracy](https://img.shields.io/badge/accuracy-≥97%25-2ea44f?style=flat-square)
![F1 Score](https://img.shields.io/badge/F1--score-≥95%25-2ea44f?style=flat-square)
![Endpoint](https://img.shields.io/badge/endpoint-POST%20%2Fpredict-orange?style=flat-square)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat-square)

![Dataset](https://img.shields.io/badge/dataset-SMS%20Spam%20Collection-blueviolet?style=flat-square)
![Vectorizer](https://img.shields.io/badge/vectorizer-TF--IDF-blueviolet?style=flat-square)
![Classifier](https://img.shields.io/badge/classifier-Naive%20Bayes%20%2F%20Logistic%20Regression-blueviolet?style=flat-square)

</div>

---

## Overview

Real-time text classification microservice that combines FastAPI with machine learning to detect spam and suspicious content. The system provides probability-based predictions through an API, making automated text analysis easy to integrate into other applications.

Every message that hits the endpoint gets scored in milliseconds, so the platform can filter malicious links and spam before they ever reach a user's inbox, with no manual review, no delay, and no gap for bad actors to slip through.

## Background

Spam and phishing links are a moving target, with new campaigns, new domains, and new tricks appearing all day long. Coral doesn't try to keep a blocklist current; it learns the *language* of spam instead. That means it catches variations a static filter would miss, while staying fast and cheap enough to sit in the critical path of every message sent on the platform.

At the target accuracy and F1 thresholds, Coral is built to catch the overwhelming majority of spam while keeping false positives (legitimate messages wrongly flagged) rare enough that users never notice the filter is there.

## How It Works

Incoming text gets vectorized (same TF-IDF representation used during training) and passed to the classifier. The response includes both the predicted label and a confidence score, so whatever's calling the API can decide how to act. it auto-block above some threshold, flag for review below it.

```json
// Request
{
  "text": "Congratulations! You've won a free $1,000 gift card. Click here to claim now."
}

// Response
{
  "prediction": "spam",
  "confidence_score": 0.98
}
```

## Performance Targets

| Metric | Target |
|---|---|
| Accuracy | ≥ 97% |
| F1-score | ≥ 95% |

## Stack

- **Model:** Naive Bayes / Logistic Regression, trained via Scikit-Learn on the SMS Spam Collection dataset
- **Features:** TF-IDF vectorization of message text
- **API:** FastAPI, with Pydantic enforcing strict request/response schemas
- **Serving:** trained model and vectorizer are serialized and loaded once at startup, so inference stays low-latency

## Coral Setup

```bash
pip install -r requirements.txt
python training/train.py        # trains and serializes the model
uvicorn app.main:app --reload   # starts the API
```

Quick Test
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hey, are we still on for lunch tomorrow?"}'
```

## Notes

- Artifacts need to be regenerated whenever the training data or preprocessing pipeline changes.
- The confidence score reflects the model's predicted probability for the returned label.
