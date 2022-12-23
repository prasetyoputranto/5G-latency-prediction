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

condition_labels = ['RSRP'] #wanted network conditions
y_label = 'rtt' #delay time in ns

#Read all parquet file in a folder
data_dir=Path('/home/pras/5g-latency-prediction/venv/parquett')#Add the path of the folder
full_df=pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)

df_train = full_df[
    [
        y_label,
        *condition_labels
    ]
]

df_train=df_train.sample(frac=0.5,random_state=1).reset_index()
df_train.rtt*=0.000001
noisertt = np.random.normal(0,0.1,len(df_train)) #add noise
df_train["rtt"]+=noisertt
df_train = df_train[
    df_train.rtt >= 0
]
print(df_train)


df_train = df_train[[y_label, *condition_labels]]
df_train["y_input"] = df_train[y_label]
df_train = df_train.drop(columns=[y_label])

batch_size = 5000
X = df_train[condition_labels]
Y = df_train.y_input
steps_per_epoch = len(df_train) // batch_size

X = np.array(X)
Y = np.array(Y)
training_data = tuple([ X[:,i] for i in range(len(condition_labels)) ]) + (Y,)

# Read all parquet files of validation dataset in a folder
data_dir=Path('/home/pras/5g-latency-prediction/venv/parquettval2')#Add the path of the folder
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

df_val.rtt*=0.000001 #convert rtt from ns into ms
print(df_val)
df_val=np.array(df_val)
validation = df_val[(df_val[:,1]==-59)]

# Gaussian modal
model = ConditionalGaussianMM(
    x_dim=condition_labels,
   centers=3,
    hidden_sizes=[8,8,8],
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
    epochs=3000,
   verbose=1,
)
x1=np.array([[-59,-59,-59],[-59,-59,-59]])
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
    centers=3,
    hidden_sizes=[8,8,8],
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
    epochs=200,
    verbose=1,
)
x_Gamma=np.array([[-59,-59,-59],[-59,-59,-59]])
y_Gamma=np.arange(0,20,0.2)
p_Gamma=np.zeros(y_Gamma.size)
for i in range(0,y_Gamma.size-1,2):
    yr_Gamma=np.array([y_Gamma[i],y_Gamma[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x_Gamma, yr_Gamma)
    p_Gamma[i]=prob[0]
    p_Gamma[i+1]=prob[1]

print(validation.size)

# Plot pdf
plt.figure(1)
plt.plot(y_Gamma, p_Gamma)
plt.plot(y, p1)
plt.xlabel('delay(ms)')
plt.ylabel('probability')
plt.legend(['Gamma mixture + Pareto model','Gaussian mixture model'], loc='upper right')
plt.hist(validation[:,0], color=['lightgreen'], density=True, bins=100)
plt.savefig("Conditional PDF on RSRP.png")

# Get dvp
o_Gamma=np.array([15]) #target delay
in_Gamma=np.arange(-90,-50,5) #RSRP=[min=-86,max=-58]
d_Gamma=np.zeros(in_Gamma.size)
for i in range(0,in_Gamma.size):
    prob, logprob, pred_cdf = model.prob_batch([[(in_Gamma[i]+90)/3]], o_Gamma)
    d_Gamma[i]=1-pred_cdf
    
# Plot dvp
plt.figure(2)
plt.plot(in_Gamma, d_Gamma)
plt.xlabel('RSRP (dBm)')
plt.ylabel('DVP')
plt.legend(['Conditional DVP for 15ms'], loc='upper right')
plt.savefig("Conditional DVP on RSRP.png")
