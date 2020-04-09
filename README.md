# classification-on-Tensorflow
Written in Python and Tensforflow 2.0
from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
import pandas as pd



# Datasets features and labels
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# to upload the datafile using Pandas

train_path = tf.keras.utils.get_file(
    'iris_training.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'
)

test_path = tf.keras.utils.get_file(
    'iris_test.csv', ' https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'
)

# To load the dataset
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names= CSV_COLUMN_NAMES, header= 0)

print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')

print(train.head)
print(train.shape)


def input_fn(features, labels, training = True, batch_size= 256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))

    # shuffle and repeat if you are in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()

        return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hiddden layers of 30 and 10 nodes respectively.
    hidden_units= [30,10],
    # The model must choose between 3 classes.
    n_classes=3)

classifier.train(
    input_fn = lambda: input_fn(train, train_y, training=True),
    steps= 5000)

eval_result= classifier.evaluate(input_fn = lambda: input_fn(test,test_y, training=False))
# print('\nTest set accuracy : { accuracy : 0.3f}\n'.format(**eval_result))

# To make predictions
def input_fn(features, batch_size = 256):
    # convert the inputs to a Dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch([batch_size])

features = ['SepalLength', 'SepalWidth', 'PetalLenght', 'PetalWidth']
predict = {}

print ('please type numeric values as prompted')
for feature in features:
    valid = True
    while valid:
        val = input(features + ':')
        if not val.isdigit():valid = False

    predict[feature]= [float(val)]

predictions = classifier.predict(input_fn = lambda : input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id],100* probability))
