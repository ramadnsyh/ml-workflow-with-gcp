import os
from google.cloud import storage
from google.oauth2 import service_account
from ast import literal_eval
from io import BytesIO, StringIO
import pickle
# from dotenv import load_dotenv

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import random_search_params as rsp

# load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
PROJECT_ID, BUCKET_NAME, PROJECT_NAME
credential = service_account.Credentials.from_service_account_info(literal_eval(os.environ.get("CREDENTIAL")))

def save_model_to_storage(gcs_path, local_path):
    client = storage.Client(project=PROJECT_NAME, credentials=credential)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path) # File path @CloudStorage
    blob.upload_from_filename(local_path) # File path @Local
    
def upload_to_gcs(gcs_path, local_path):
    client = storage.Client(project=PROJECT_NAME, credentials=credential)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path) # File path @CloudStorage
    blob.upload_from_filename(local_path) # File path @Local
    
def read_file_from_gcs(gcs_path):
    client = storage.Client(project=PROJECT_NAME, credentials=credential)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(gcs_path) # File path @CloudStorage
    # blob.download_to_filename("data/diabetes_gcs.csv") # Download to local
    df = pd.read_csv(BytesIO(blob.download_as_string()))
    return df
  
def load_model_from_storage(model_path):
    client = storage.Client(project=PROJECT_NAME, credentials=credential)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(model_path) # File path @CloudStorage
    # blob.download_to_filename("data/diabetes_gcs.csv") # Download to local
    gcs_model = pickle.load(BytesIO(blob.download_as_string()))
    return gcs_model
  
def save_result_to_storage(df, file_name):
    df_new = df.to_csv(index=False)
    content_type = "text/csv" if ".csv" in file_name else "text/plain"
    client = storage.Client(project=PROJECT_NAME, credentials=credential)
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(file_name) # File path @CloudStorage
    blob.upload_from_string(df_new, content_type=content_type)
    
def predict(request):
    df = read_file_from_gcs("input/diabetes.csv")
    model = load_model_from_storage("model/model.pkl")
    df["prediction"] = model.predict(df)
    save_result_to_storage(df, "output/result_from_cloudfunction.csv")
    pass