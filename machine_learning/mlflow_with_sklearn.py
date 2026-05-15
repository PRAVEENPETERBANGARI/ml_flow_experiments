# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Cell 1
# import requests
# import base64

# # Parameters
# git_raw_url = "https://raw.githubusercontent.com/PRAVEENPETERBANGARI/ml_flow_experiments/main/dataset/diabetes.csv"
# volume_path = "/Volumes/mlflow_catalog/ml_experiments_schema/dataset/diabetes.csv"

# # Download CSV from GitHub
# response = requests.get(git_raw_url)
# response.raise_for_status()
# csv_content = response.content

# # Write to Unity Catalog volume
# with open(f"{volume_path}", "wb") as f:
#     f.write(csv_content)

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

volume_path = "/Volumes/mlflow_catalog/ml_experiments_schema/dataset/diabetes.csv"
df = spark.read.format("csv").option("header", "true").load(volume_path)
display(df)

# COMMAND ----------

double_cols = ["BMI", "DiabetesPedigreeFunction"]
int_cols = [c for c in df.columns if c not in double_cols]

final_df = df.select(*[col(c).cast(IntegerType()).alias(c) for c in int_cols] 
                    + [col(c).cast(DoubleType()).alias(c) for c in double_cols])
display(final_df)


# COMMAND ----------

import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

mlflow.set_experiment(f"{os.getcwd()}/diabetes_predictor")

pandas_df = final_df.toPandas()
X = pandas_df.drop("Outcome", axis=1)
y = pandas_df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Diabetes") as run:
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=4,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # mlflow.log_metric("accuracy", acc)
    # mlflow.log_param("n_estimators", 100)
    # mlflow.log_param("max_depth", 5)
    # mlflow.log_param("min_samples_split", 4)
    # mlflow.sklearn.log_model(
    #     clf,
    #     name="random_forest_model"
    # )

# COMMAND ----------

model_url = f"runs:/{run.info.run_id}/model"
print(f"Model URL: {model_url}")
register_model = mlflow.register_model(model_url, "mlflow_catalog.ml_experiments_schema.diabetes_predictor")

# COMMAND ----------


