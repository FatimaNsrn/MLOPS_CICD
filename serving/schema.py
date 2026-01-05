from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("file:///E:/MLOPS/mlruns")  # your correct backend path
client = MlflowClient()

versions = client.search_model_versions("name='profit_classifier_pipeline'")
for v in versions:
    print("Version:", v.version, "Aliases:", v.aliases)
