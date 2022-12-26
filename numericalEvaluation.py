import pandas as pd
import numpy
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
os.environ['CUDA_VISIBLE_DEVICES'] = ""
from pathlib import Path
from pr3d.de import ConditionalGaussianMM
from pr3d.de import ConditionalGammaMixtureEVM


condition_labels = ['X','Y'] #wanted network conditions
y_label = 'rtt' #delay time in ns

#Read all parquet file in a folder
data_dir=Path('/home/pras/5g-latency-prediction/venv/parquett')#Add the path of the folder
full_df=pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob('*.parquet')
)
# print(full_df)
df_train = full_df[
    [
        y_label,
        *condition_labels
    ]
]
# print(df_train)
df_train=df_train.sample(frac=0.5,random_state=1).reset_index()
df_train.Y*=2
df_train.X*=2
df_train.rtt*=0.000001

noisertt = np.random.normal(0,1,len(df_train)) #add noise
df_train["rtt"]+=noisertt

df_train = df_train[[y_label, *condition_labels]]
df_train["y_input"] = df_train[y_label]
df_train = df_train.drop(columns=[y_label])

batch_size = 50000
X = df_train[condition_labels]
Y = df_train.y_input
steps_per_epoch = len(df_train) // batch_size

X = np.array(X)
Y = np.array(Y)
training_data = tuple([ X[:,i] for i in range(len(condition_labels)) ]) + (Y,)

# Validation
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
df_val.Y*=2
df_val.X*=2
df_val.rtt*=0.000001
#noiserttv = np.random.normal(0,1,len(df_val)) #add noise
#df_val["rtt"]+=noiserttv
df_val=np.array(df_val)
validation = df_val[(df_val[:,1]==8) & (df_val[:,2]==5)]

# Gaussian modal
model = ConditionalGaussianMM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)
model.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.05,
    ),
    loss=model.loss,
)
model.training_model.fit(
    x=training_data,
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=1000,
    verbose=1,
)

x1=np.array([[8,5],[8,5]])
y=np.arange(7,22,0.2)
p1=np.zeros(y.size)
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x1, yr)
    p1[i]=1-pred_cdf[0]
    p1[i+1]=1-pred_cdf[1]

# Evaluation
ye=np.array([15])

validation = df_val[(df_val[:,1]==8) & (df_val[:,2]==5)]
val1=np.sum(validation[:,0]>=15)/validation[:,0].size
prob, logprob, pred_cdf = model.prob_batch([[8,5]], ye)
dvp1=1-pred_cdf
valp1=np.zeros(y.size)
for i in range(0,y.size-1):
    valp1[i]=np.sum(validation[:,0]>=y[i])/validation[:,0].size


validation2 = df_val[(df_val[:,1]==1) & (df_val[:,2]==0)]
val2=np.sum(validation2[:,0]>=15)/validation2[:,0].size
prob, logprob, pred_cdf = model.prob_batch([[1,0]], ye)
dvp2=1-pred_cdf

validation3 = df_val[(df_val[:,1]==4) & (df_val[:,2]==5)]
val3=np.sum(validation3[:,0]>=15)/validation3[:,0].size
prob, logprob, pred_cdf = model.prob_batch([[4,5]], ye)
dvp3=1-pred_cdf

# Numerical Evaluation
table = PrettyTable(['validation position','target delay','dvp from dataset','dvp from model'])
table.add_row(['[8,5]','15ms',val1,dvp1])
table.add_row(['[4,5]','15ms',val3,dvp3])
table.add_row(['[1,0]','15ms',val2,dvp2])
print(table)

# Graphical Evaluation
plt.plot(y,p1,'r--',label='DVP from model')
plt.plot(y,valp1,'b--',label='DVP from dataset')
plt.legend()
plt.xlabel('target delay(ms)')
plt.ylabel('DVP')
plt.savefig("dvpevaluation.png") 
