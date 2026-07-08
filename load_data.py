import kagglehub
import os

os.environ['KAGGLEHUB_CACHE'] = 'dataset'
path = kagglehub.dataset_download("moltean/fruits")

print("Path to dataset files:", path)