FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn>=1.4.0 \
    pandas>=2.0.0 \
    numpy>=1.26.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    fastapi \
    uvicorn

# Copy model artifacts and dataset
COPY mlruns/ /app/mlruns/
COPY MLProject/heart_disease_preprocessing.csv /app/heart_disease_preprocessing.csv

# Set MLflow tracking URI to local mlruns
ENV MLFLOW_TRACKING_URI=/app/mlruns
ENV MODEL_URI=mlruns

EXPOSE 8080

# Serve model using MLflow built-in server
CMD ["sh", "-c", "RUN_ID=$(python -c \"import mlflow; client=mlflow.MlflowClient(); exp=client.get_experiment_by_name('Heart Disease - CI'); runs=client.search_runs(exp.experiment_id, order_by=['start_time DESC'], max_results=1); print(runs[0].info.run_id)\") && mlflow models serve -m runs:/$RUN_ID/model --host 0.0.0.0 --port 8080 --env-manager local"]
