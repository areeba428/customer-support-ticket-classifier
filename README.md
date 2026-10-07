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

It also logs the metrics and registers a complete raw-text inference pipeline
as `customer-support-ticket-classifier` in MLflow. By default, MLflow metadata
is stored in `mlflow.db` (SQLite) and model files in `mlartifacts/`.

Start the local MLflow UI after training:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Then open [http://127.0.0.1:5001](http://127.0.0.1:5001) and select
**Models** to see the registered version.

To use a remote MLflow tracking server or a different model name:

```bash
export MLFLOW_TRACKING_URI=https://your-mlflow-server
export MLFLOW_REGISTERED_MODEL_NAME=customer-support-ticket-classifier
python train_ticket_classifier.py
```

## Deploy with Jenkins

This follows the same pattern as a Pipeline job that checks out a GitHub repo and runs `Jenkinsfile`.

1. Push this project to GitHub. Include `data/customer_support_tickets.csv` and `model/`, because the pipeline trains from that CSV and the image serves the saved `.pkl` files.
2. In Jenkins, choose **New Item**, name it `ticket-classifier`, and pick **Pipeline**.
3. Under **Pipeline**, select **Pipeline script from SCM**, set the repository URL, branch `*/main`, and script path `Jenkinsfile`.
4. Save, then click **Build Now**.

The pipeline:

1. Checks out the code
2. Trains the model and registers a new version in MLflow. The registry is stored at `/var/lib/jenkins/mlflow/customer-support-ticket-classifier/` so it stays after the build workspace is cleaned.
3. Builds the `ticket-classifier` Docker image
4. Replaces the running `ticket-classifier` container and publishes port **5000**
5. Calls `http://127.0.0.1:5000/health` until the app responds

The Jenkins user must be allowed to run Docker. On this machine that user is already in the `docker` group.

Open the app at [http://127.0.0.1:5000](http://127.0.0.1:5000) after the build is green. If port 5000 is already taken by another container, stop that container before building.

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
| `Dockerfile` | Image used by Jenkins to run the Flask app |
| `Jenkinsfile` | Jenkins pipeline: train, register, build, deploy |
