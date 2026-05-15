# Databricks notebook source
# import mlflow
# import mlflow.spark
# import optuna
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# def objective(trial):
#     numTrees = trial.suggest_int("numTrees", 10, 100)
#     maxDepth = trial.suggest_int("maxDepth", 3, 10)
#     rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=numTrees, maxDepth=maxDepth)
#     stages = indexers + encoders + [assembler, rf]
#     pipeline = Pipeline(stages=stages)
#     paramGrid = ParamGridBuilder().build()
#     cv = CrossValidator(
#         estimator=pipeline,
#         estimatorParamMaps=paramGrid,
#         evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"),
#         numFolds=3
#     )
#     with mlflow.start_run(nested=True):
#         model = cv.fit(train_df)
#         predictions = model.transform(test_df)
#         auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions)
#         mlflow.log_param("numTrees", numTrees)
#         mlflow.log_param("maxDepth", maxDepth)
#         mlflow.log_metric("AUC", auc)
#         mlflow.spark.log_model(model.bestModel, "model")
#     return auc

# mlflow.set_experiment("/Users/your_user/RandomForestOptuna")
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=10)

# best_params = study.best_params
# rf_best = RandomForestClassifier(labelCol="label", featuresCol="features", **best_params)
# stages_best = indexers + encoders + [assembler, rf_best]
# pipeline_best = Pipeline(stages=stages_best)
# with mlflow.start_run(run_name="BestModel"):
#     model_best = pipeline_best.fit(train_df)
#     mlflow.spark.log_model(model_best, "model")
#     predictions_best = model_best.transform(test_df)
#     auc_best = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(predictions_best)
#     mlflow.log_metric("AUC", auc_best)
#     display(predictions_best.select("label", "prediction", "probability"))

# COMMAND ----------


from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer,
    OneHotEncoder
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# COMMAND ----------

# Read data
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/path/to/data.csv")

# Identify categorical and numerical columns
categorical_cols = [field.name for field in df.schema.fields if field.dataType == "StringType" and field.name != "label"]
numerical_cols = [field.name for field in df.schema.fields if field.dataType != "StringType" and field.name != "label"]

# StringIndexer and OneHotEncoder for categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_ohe") for col in categorical_cols]

# Assemble features
assembler_inputs = [f"{col}_ohe" for col in categorical_cols] + numerical_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Random Forest Classifier
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Pipeline
stages = indexers + encoders + [assembler, rf]
pipeline = Pipeline(stages=stages)

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train model
model = pipeline.fit(train_df)

# Predict
predictions = model.transform(test_df)

# Evaluation
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

# Display metrics
display(predictions.select("label", "prediction", "probability"))
print(f"AUC: {auc}")
