from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import os

os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['hadoop.home.dir'] = 'C:\\hadoop'

spark = SparkSession.builder \
    .appName("DiabetesTraining") \
    .getOrCreate()

df = spark.read.csv('offline.csv', header=True, inferSchema=True)

feature_cols = [col for col in df.columns if col != 'Diabetes_binary']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="Diabetes_binary",
    predictionCol="prediction",
    metricName="f1"
)

models = []

print("Training Logistic Regression...")
lr = LogisticRegression(featuresCol="features", labelCol="Diabetes_binary")
lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
lr_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()
lr_cv = CrossValidator(estimator=lr_pipeline, estimatorParamMaps=lr_grid,
                       evaluator=f1_evaluator, numFolds=3)
lr_model = lr_cv.fit(train_data)
lr_f1 = f1_evaluator.evaluate(lr_model.transform(test_data))
models.append(("LogisticRegression", lr_model, lr_f1))
print(f"LR F1: {lr_f1:.4f}")

print("Training Random Forest...")
rf = RandomForestClassifier(featuresCol="features", labelCol="Diabetes_binary")
rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
rf_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()
rf_cv = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=rf_grid,
                       evaluator=f1_evaluator, numFolds=3)
rf_model = rf_cv.fit(train_data)
rf_f1 = f1_evaluator.evaluate(rf_model.transform(test_data))
models.append(("RandomForest", rf_model, rf_f1))
print(f"RF F1: {rf_f1:.4f}")

print("Training GBT...")
gbt = GBTClassifier(featuresCol="features", labelCol="Diabetes_binary")
gbt_pipeline = Pipeline(stages=[assembler, scaler, gbt])
gbt_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [50, 100]) \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .build()
gbt_cv = CrossValidator(estimator=gbt_pipeline, estimatorParamMaps=gbt_grid,
                        evaluator=f1_evaluator, numFolds=3)
gbt_model = gbt_cv.fit(train_data)
gbt_f1 = f1_evaluator.evaluate(gbt_model.transform(test_data))
models.append(("GBT", gbt_model, gbt_f1))
print(f"GBT F1: {gbt_f1:.4f}")

best_name, best_cv_model, best_f1 = max(models, key=lambda x: x[2])
print(f"\nBest model: {best_name} with F1={best_f1:.4f}")

best_model = best_cv_model.bestModel
best_model.write().overwrite().save("best_model")

spark.stop()