import pandas as pd
import numpy
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = ""

from pr3d.de import ConditionalGaussianMM
from pr3d.de import ConditionalGammaMixtureEVM
condition_labels = ['queue_length_h0','queue_length_h1']
y_label = 'end2end_delay'

df_train = pd.read_parquet('dataset.parquet')
df_train = df_train[
    [
        y_label,
        *condition_labels
    ]
]
df_train = df_train[
    df_train.queue_length_h0 >= 0
]
df_train = df_train[
    df_train.queue_length_h1 >= 0
]
df_train2=np.array(df_train)
train = df_train2[(df_train2[:,1]==1) & (df_train2[:,2]==5)]
df_train = df_train[[y_label, *condition_labels]]
df_train["y_input"] = df_train[y_label]
df_train = df_train.drop(columns=[y_label])

# Gaussian modal
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
        learning_rate=0.0002,
    ),
    loss=model.loss,
)
X = np.array(X)
Y = np.array(Y)
training_data = tuple([ X[:,i] for i in range(len(condition_labels)) ]) + (Y,)
model.training_model.fit(
    x=training_data,
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=2500,
    verbose=1,
)
x1=np.array([[1,5],[1,5]])
y=np.arange(250)
p1=np.zeros(y.size)
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x1, yr)
    p1[i]=prob[0]
    p1[i+1]=prob[1]

# Gamma modal
model = ConditionalGammaMixtureEVM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)

model.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.002,
    ),
    loss=model.loss,
)
model.training_model.fit(
    x=training_data,
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=1500,
    verbose=1,
)
x_Gamma=np.array([[1,5],[1,5]])
y_Gamma=np.arange(250)
p_Gamma=np.zeros(y_Gamma.size)
for i in range(0,y_Gamma.size-1,2):
    yr_Gamma=np.array([y_Gamma[i],y_Gamma[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x_Gamma, yr_Gamma)
    p_Gamma[i]=prob[0]
    p_Gamma[i+1]=prob[1]

print(train)
# Plot
plt.plot(y_Gamma, p_Gamma)
plt.plot(y, p1)
plt.xlabel('delay')
plt.ylabel('probability')
plt.legend(['Gamma', 'Gaussian'], loc='upper right')
plt.hist(train[:,0], color=['lightgreen'], density=True, bins=60)
