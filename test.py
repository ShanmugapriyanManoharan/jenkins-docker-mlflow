import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

def get_highest_model_version(model_name: str):
    mlflow.set_tracking_uri(uri="http://localhost:8070")
    client = MlflowClient()
    
    try:
        # fetch al versions for the model_name
        all_versions = client.search_model_versions(f"name= '{model_name}'")
        if not all_versions:
            print(f"No versions found for model '{model_name}'.")
            return None
            
        latest_version = max(all_versions, key=lambda v:int(v.version))
        
        print(f"Highest model version of '{model_name}':")
        print(f"Version: '{latest_version.version}':")
        print(f"Stage: '{latest_version.current_stage}':")
        print(f"Run ID: '{latest_version.run_id}':")
        print(f"Artifact URI: '{latest_version.source}':")
        
        return latest_version.version
        
    except Exception as e:
        print(f"Error fetching model version: {e}")
        return None

def model_prediction(model_name, version):
    mlflow.set_tracking_uri(uri="http://host.docker.internal:8070")
    # Load your saved model from MLflow
    model_uri = f"models:/{model_name}/{version}"  # Replace <registered_model_name> with your model name
    model = mlflow.pyfunc.load_model(model_uri)

    # Define a sample input (e.g., from your test dataset)
    # Ensure the input format is the same as what your model expects
    sample_input = pd.DataFrame({
        "age": [57],
        "sex": [1],
        "cp": [0],
        "trestbps": [110],
        "chol": [201],
        "fbs": [0],
        "restecg": [1],
        "thalach": [126],
        "exang": [1],
        "oldpeak": [1.5],
        "slope": [1],
        "ca": [0],
        "thal": [1]
    })


    # Make a prediction using the loaded model
    prediction = model.predict(sample_input)

    # Print the prediction result
    print("Prediction for the sample input:", prediction)
    print("Test Completed")

if __name__ == "__main__":
    model_name = "heart_disease"
    
    version = get_highest_model_version(model_name)
    print(f"Model Name: {model_name} --> Version: {version}")
    
    model_prediction(model_name, version)