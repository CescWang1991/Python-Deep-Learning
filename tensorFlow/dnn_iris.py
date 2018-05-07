import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

np.random.seed(42)
data = load_iris()
feature = data["data"]
label = data["target"]

X_train, X_test, y_train, y_test = train_test_split(feature, label, random_state=42, test_size=0.2)
#estimator = tf.contrib.estimator.DNNEstimator(
#    feature_columns=data["feature_names"],
#    head=tf.contrib.estimator.multi_label_head(n_classes=3),
#    hidden_units=[1024, 512, 256])
print(dict(X_train))
#dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))
#dataset = dataset.shuffle(1000).repeat().batch(X_train.shape[0])
#print(dataset)