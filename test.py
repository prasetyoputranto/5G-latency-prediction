import pandas as pd
import numpy
import os
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ""

from pr3d.de import ConditionalGaussianMM

condition_labels = ['longer_delay_prob_h0'] #,'longer_delay_prob_h1','queue_length_h0','queue_length_h1']
y_label = 'end2end_delay'

df_train = pd.read_parquet('dataset.parquet')
df_train = df_train[
    [
        y_label,
        *condition_labels
    ]
]
#df_train = df_train[
#    df_train.queue_length_h0 >= 0
#]
#df_train = df_train[
#    df_train.queue_length_h1 >= 0
#]
# dataset pre process
df_train = df_train[[y_label, *condition_labels]]
df_train["y_input"] = df_train[y_label]
df_train = df_train.drop(columns=[y_label])

print(df_train)
print("Hello world!")

model = ConditionalGaussianMM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)

batch_size = 1000

X = df_train[condition_labels]
Y = df_train.y_input

steps_per_epoch = len(df_train) // batch_size

model.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
    ),
    loss=model.loss,
)

model.training_model.fit(
    x=[X, Y],
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    verbose=1,
)

x=np.array([[4],[4],[4],[4],[4],[4]])
y=np.array([20,40,60,80,100,120])

prob, logprob, pred_cdf = model.prob_batch(x, y)

print(prob)
print(logprob)