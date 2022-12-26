import pandas as pd
import math
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

noisertt = np.random.normal(0,0.1,len(df_train)) #add noise
df_train["rtt"]+=noisertt

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

# Validation
data_dir=Path('//home/pras/5g-latency-prediction/venv/parquettval2')#Add the path of the folder
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
df_val=np.array(df_val)
validation = df_val[(df_val[:,1]==1) & (df_val[:,2]==0)]

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
# Gamma modal
model2 = ConditionalGammaMixtureEVM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)
model2.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.01,
    ),
    loss=model.loss,
)
model2.training_model.fit(
    x=training_data,
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=800,
    verbose=1,
)


x1=np.array([[4,5],[4,5]])
y=np.arange(7,57,0.5)
p1=np.zeros(y.size)
p2=np.zeros(y.size)
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob, logprob, pred_cdf = model.prob_batch(x1, yr)
    p1[i]=1-pred_cdf[0]
    p1[i+1]=1-pred_cdf[1]
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob2, logprob2, pred_cdf2 = model2.prob_batch(x1, yr)
    p2[i]=1-pred_cdf2[0]
    p2[i+1]=1-pred_cdf2[1]    
# Evaluation
ye=np.array([10])

validation = df_val[(df_val[:,1]==1) & (df_val[:,2]==0)]
valp1=np.zeros(y.size)
for i in range(0,y.size-1):
    valp1[i]=np.sum(validation[:,0]>=y[i])/validation[:,0].size

# Substract dvp for model from dvp for validation dataset
d1=np.zeros(y.size)
d2=np.zeros(y.size)
d1=abs(p1-valp1) # substract
d2=abs(p2-valp1)
lp1=numpy.log10(p1+1e-10)
lp2=numpy.log10(p2+1e-10)
lvalp1=numpy.log10(valp1+1e-10)
ld1=abs(lp1-lvalp1)
ld2=abs(lp2-lvalp1)


# Plot
# log
plt.figure(1)
plt.plot(y,p1,'r--',label='dvp from Gaussian mixture model')
plt.plot(y,valp1,'b--',label='dvp from dataset')
plt.plot(y,p2,'g--',label='dvp from Gamma mixture + Pareto model')
plt.legend()
plt.xlim([7,20])
plt.title('dvp comparison for validation point [4,5]')
plt.xlabel('target delay/ms')
plt.ylabel('dvp')
plt.savefig("dvp comparison.png") 

plt.figure(2)
plt.plot(y,p1,'r--',label='dvp from Gaussian mixture model')
plt.plot(y,valp1,'b--',label='dvp from dataset')
plt.plot(y,p2,'g--',label='dvp from Gamma mixture + Pareto model')
plt.yscale('log') # turn y axis to log
plt.legend()
plt.xlim([7,20])
plt.title('log dvp comparison for validation point [4,5]')
plt.xlabel('target delay/ms')
plt.ylabel('dvp')
plt.savefig("log dvp comparison.png")

plt.figure(3)
plt.plot(y,ld1,'r--',marker ='.',label='log dvp error of Gaussian mixture model')
plt.plot(y,ld2,'g--',marker ='.',label='log dvp error of Gamma mixture + Pareto model')
plt.legend()
plt.xlabel('target delay/ms')
plt.ylabel('log dvp error')
plt.title('Log dvp error for validation point [4,5]')
plt.ylim([0,4])
plt.xlim([7,20])
plt.savefig("log dvp error.png") 
