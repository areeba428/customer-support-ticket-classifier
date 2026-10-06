# Customer Support Ticket Classifier

Train a scikit-learn **Logistic Regression** model to classify customer-support
tickets by **Ticket Type**, then serve predictions with a **Flask** web app.

## Dataset

- Source: [Customer Support Ticket Dataset on Kaggle](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)
- Local file: `data/customer_support_tickets.csv`
- Input feature: `Ticket Description`
- Target label: `Ticket Type`
- Not used as inputs: `Ticket Type`, `Resolution`

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_ticket_classifier.py
```

This saves:

- `model/tfidf_vectorizer.pkl`
- `model/logistic_regression_model.pkl`

## Run the Flask web app

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

You can also call the JSON API:

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"I was charged twice on my last invoice.\"}"
```

## Project files

| File / folder | Purpose |
|---|---|
| `data/customer_support_tickets.csv` | Kaggle dataset |
| `train_ticket_classifier.py` | Train and evaluate the model |
| `model/*.pkl` | Saved TF-IDF vectorizer and Logistic Regression model |
| `app.py` | Flask web application |
| `templates/index.html` | Web page UI |
| `static/style.css` | Page styling |
