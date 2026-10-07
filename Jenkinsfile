pipeline {
    agent any

    environment {
        IMAGE_NAME = 'ticket-classifier'
        CONTAINER_NAME = 'ticket-classifier'
        MLFLOW_DIR = '/var/lib/jenkins/mlflow/customer-support-ticket-classifier'
        MLFLOW_TRACKING_URI = 'sqlite:////var/lib/jenkins/mlflow/customer-support-ticket-classifier/mlflow.db'
        MLFLOW_ARTIFACT_LOCATION = 'file:///var/lib/jenkins/mlflow/customer-support-ticket-classifier/mlartifacts'
        MLFLOW_EXPERIMENT_NAME = 'customer-support-ticket-classifier'
        MLFLOW_REGISTERED_MODEL_NAME = 'customer-support-ticket-classifier'
    }

    options {
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Train and register model') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    mkdir -p "$MLFLOW_DIR"
                    python train_ticket_classifier.py
                '''
            }
        }

        stage('Build Docker image') {
            steps {
                sh 'docker build -t "$IMAGE_NAME" .'
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                    docker rm -f "$CONTAINER_NAME" || true
                    docker run -d -p 5000:5000 --name "$CONTAINER_NAME" "$IMAGE_NAME"
                '''
            }
        }

        stage('Smoke test') {
            steps {
                sh '''
                    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
                        if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2).read()"; then
                            exit 0
                        fi
                        sleep 2
                    done
                    echo "App did not become healthy"
                    docker logs "$CONTAINER_NAME" || true
                    exit 1
                '''
            }
        }
    }

    post {
        always {
            echo 'Jenkins pipeline finished.'
        }
    }
}
