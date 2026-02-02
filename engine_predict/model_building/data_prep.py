# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi



def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return filtered_df

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/vihu21/predictive_maintenance/engine_predict/master/data/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")


# Define target variable
target_col = 'Engine Condition'
# Example usage
df_clean = remove_outliers_iqr(df, 'Engine rpm')
# Split into X (features) and y (target)
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

#X = df.drop(columns=[target_col])
#y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

repo_name = ""

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"engine_predict/master/data/{file_path}",  # just the filename
        repo_id="vihu21/predictive_maintenance",
        repo_type="dataset",
    )
