###############################################YOU CAN NOT USE THIS CODE WITHOUT PERMISSION ################################

# -*- coding:utf-8 -*-
#firstApp.py
# if __name__=="__main__":
#     from pyspark.sql import SparkSession
#     sc=SparkContext(appName='firt app')
#     spark = SparkSession.builder.setMaster("local[*]").appName("test").getOrCreate()
#1.load data
df = spark.read.parquet("hdfs://namenode:9000/dis_materials/fast2_metadata.parquet").sample(fraction=0.2, seed=3)


df = df.repartition(1000)
#2. mapping data
mapping = {col: col.replace('.','_').replace(' ', '_') for col in df.columns} 
from pyspark.sql.functions import col
df = df.select([col('`'+c+'`').alias(mapping.get(c, c)) for c in df.columns])

#3.process features and label
#3.1 categorical feature
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
# filter categorical columns
categoricalColumns = [str(t[0]) for t in df.dtypes if ((t[1] == 'string' ) and t[0].find('in_') == 0) and t[0].find('in_cooling_fuel') == -1 and t[0].find('in_heating_fuel') == -1 and t[0].find('in_hvac_delivery_type') == -1]

# use StringIndexer and OneHotEncoderEstimator to process categorical columns 
# define stages
stages = []
for col in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = col, outputCol = col + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[col + "classVec"])
    stages += [stringIndexer, encoder]

#3.2 process numeric columns
# add numeric columns
numericCols = [str(t[0])for t in df.dtypes if ((t[1] == 'double' or t[1] == 'long' or t[1] == 'boolean') and t[0].find('in_') == 0)]
cols = categoricalColumns + numericCols + ['out_electricity_total_energy_consumption']

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# 4. split to two data set
from pyspark.ml import Pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
SelCol = ['out_electricity_total_energy_consumption', 'features']
df = df.select(SelCol)
df.printSchema()
#split train & test data
train, test = df.randomSplit([0.7, 0.3], seed = 666)

#5.train model
## 5.1.  LinearRegression  ##
from pyspark.ml.regression import LinearRegression
LR = LinearRegression(featuresCol = 'features', labelCol = 'out_electricity_total_energy_consumption', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = LR.fit(train)
predictions_LR = lrModel.transform(test)
predictions_LR.select("prediction", "out_electricity_total_energy_consumption", "features").show(10)

## 5.2.  DecisionTreeRegressor ##
from pyspark.ml.regression import DecisionTreeRegressor
DT = DecisionTreeRegressor(featuresCol = 'features', labelCol = 'out_electricity_total_energy_consumption', maxDepth = None)
dtModel = DT.fit(train)
predictions_DT = dtModel.transform(test)
predictions_DT.select("prediction", "out_electricity_total_energy_consumption", "features").show(10)


## 5.3.   RandomForestRegressor ##
from pyspark.ml.regression import RandomForestRegressor
RF = RandomForestRegressor(featuresCol = 'features', labelCol = 'out_electricity_total_energy_consumption', numTrees=200, maxDepth = None)
rfModel = RF.fit(train)
predictions_RF = rfModel.transform(test)
predictions_RF.select( "prediction", "out_electricity_total_energy_consumption", "features").show(10)


# 5.4.  GBTRegressor ##
from pyspark.ml.regression import GBTRegressor
GBT = GBTRegressor(maxIter=10, labelCol = 'out_electricity_total_energy_consumption', maxDepth = None)
gbtModel = GBT.fit(train)
predictions_GBT = gbtModel.transform(test)
predictions_GBT.select("prediction", "out_electricity_total_energy_consumption", "features").show(10)

# 6. performace
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator( labelCol = 'out_electricity_total_energy_consumption')
print('RMSE: Logistic Regression   : ', evaluator.evaluate(predictions_LR))
print('RMSE: Decision Tree         : ', evaluator.evaluate(predictions_DT))
print('RMSE: Random Forest         : ', evaluator.evaluate(predictions_RF))
print('RMSE: Gradient-Boosted Tree : ', evaluator.evaluate(predictions_GBT))

