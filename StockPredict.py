import csv
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import numpy as np
import math
from tensorflow.python.data import Dataset
array = [["Day1", "Day2", "Day3", "Day4", "Day5", "Target"]]
array.append([])
DAY_OFFSET = 1
with open('PLUG.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    i = 0
    dayI = 0
    week = 1
    for row in reader:
        if i > DAY_OFFSET:
            array[week].append(row[3])
            dayI += 1
            if dayI == 5:
                array[week].append(array[week-1][0])
                week += 1
                dayI = 0
                array.append([])

        i += 1
print(array)
del array[1]
print(array)
with open('PLUG-FORMATTED.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for row in array:
        writer.writerow(row)

stock_dataframe = pd.read_csv("PLUG-FORMATTED.csv", sep=",")
features = stock_dataframe[["Day1", "Day2", "Day3", "Day4", "Day5", "Target"]]
features = features[:len(features)-1]
print(features)
feature_columns = [tf.feature_column.numeric_column("Day1"),
                   tf.feature_column.numeric_column("Day2"),
                   tf.feature_column.numeric_column("Day3"),
                   tf.feature_column.numeric_column("Day4"),
                   tf.feature_column.numeric_column("Day5")]
targets = stock_dataframe["Target"]
feature_columns=feature_columns[:len(feature_columns)-1]
targets = targets[:len(targets)-1]
print(targets)
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

_ = linear_regressor.train(input_fn=lambda:my_input_fn(features, targets), steps=1000)

predictions_fn = lambda : my_input_fn(features, targets, num_epochs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=predictions_fn)
predictions = np.array([item['predictions'][0] for item in predictions])
print(predictions)
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean squared Error on training data %0.3f" % mean_squared_error)
print("Root mean squared error on training data $0.3f" % root_mean_squared_error)