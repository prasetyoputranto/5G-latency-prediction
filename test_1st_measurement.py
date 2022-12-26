import pandas as pd
import numpy
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



os.environ['CUDA_VISIBLE_DEVICES'] = ""
from pathlib import Path
from pr3d.de import ConditionalGaussianMM
from pr3d.de import ConditionalGammaMixtureEVM


condition_labels = ['X','Y'] #wanted network conditions
y_label = 'rtt' #delay time in ns

#Read all parquet files of training dataset in a folder
data_dir=Path('/home/pras/5g-latency-prediction/venv/parquett')#Add the path of the folder
full_df=pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)

#Adding noise mean=0 std=1ms to training data
noisertt = np.random.normal(0,1000000,len(full_df))
full_df.rtt+=noisertt

df_train = full_df[
    [
        y_label,
        *condition_labels
    ]
]

df_train = df_train[
    df_train.rtt >= 0
]
print(df_train)
df_train=df_train.sample(frac=0.6,random_state=1).reset_index() #randomly sample 60% of the dataset
df_train.Y*=2 #convert coordinate into meter unit
df_train.X*=2 #convert coordinate into meter unit
df_train.rtt*=0.000001 #convert rtt from ns into ms
df_train = df_train[[y_label, *condition_labels]]
df_train["y_input"] = df_train[y_label]
df_train = df_train.drop(columns=[y_label])

batch_size = 100000
X = df_train[condition_labels]
Y = df_train.y_input
steps_per_epoch = len(df_train) // batch_size

X = np.array(X)
Y = np.array(Y)
training_data = tuple([ X[:,i] for i in range(len(condition_labels)) ]) + (Y,)

# Read all parquet files of validation dataset in a folder
data_dir=Path('/home/pras/5g-latency-prediction/venv/parquettval')#Add the path of the folder
val_df=pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)

df_val = val_df[
    [
        y_label,
        *condition_labels
    ]
]

df_val.Y*=2 #convert coordinate into meter unit
df_val.X*=2 #convert coordinate into meter unit
df_val.rtt*=0.000001 #convert rtt from ns into ms
print(df_val)
df_val=np.array(df_val)
validation = df_val[(df_val[:,1]==8) & (df_val[:,2]==5)]

# Gaussian modal
model = ConditionalGaussianMM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[7,7,7],
    dtype="float32",
)
model.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01,
    ),
    loss=model.loss,
)
model.training_model.fit(
    x=training_data,
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=7000,
    verbose=1,
)
x1=np.array([[8,5],[8,5]])
y=np.arange(0,20,0.2)
p1=np.zeros(y.size)
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x1, yr)
    p1[i]=prob[0]
    p1[i+1]=prob[1]

# Gamma modal
model = ConditionalGammaMixtureEVM(
    x_dim=condition_labels,
    centers=4,
    hidden_sizes=[8,8,8,8],
    dtype="float32",
)

model.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.005,
    ),
    loss=model.loss,
)
model.training_model.fit(
    x=training_data,
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=7000,
    verbose=1,
)
x_Gamma=np.array([[8,5],[8,5]])
y_Gamma=np.arange(0,20,0.2)
p_Gamma=np.zeros(y_Gamma.size)
for i in range(0,y_Gamma.size-1,2):
    yr_Gamma=np.array([y_Gamma[i],y_Gamma[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x_Gamma, yr_Gamma)
    p_Gamma[i]=prob[0]
    p_Gamma[i+1]=prob[1]

print(validation.size)

# Plot
plt.plot(y_Gamma, p_Gamma)
plt.plot(y, p1)
plt.xlabel('delay')
plt.ylabel('probability')
plt.legend(['Gamma', 'Gaussian'], loc='upper right')
plt.hist(validation[:,0], color=['lightgreen'], density=True, bins=100)
