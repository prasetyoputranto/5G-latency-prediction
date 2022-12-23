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

df_train = full_df[
    [
        y_label,
        *condition_labels
    ]
]


# frac=1
df_train1=df_train.sample(frac=1,random_state=1).reset_index()
df_train1.Y*=2
df_train1.X*=2
df_train1.rtt*=0.000001

noisertt1 = np.random.normal(0,1,len(df_train1)) #add noise
df_train1["rtt"]+=noisertt1

df_train1 = df_train1[[y_label, *condition_labels]]
df_train1["y_input"] = df_train1[y_label]
df_train1 = df_train1.drop(columns=[y_label])

batch_size1 = 10000
X1 = df_train1[condition_labels]
Y1 = df_train1.y_input
steps_per_epoch1 = len(df_train) // batch_size1

X1 = np.array(X1)
Y1 = np.array(Y1)
training_data1 = tuple([ X1[:,i] for i in range(len(condition_labels)) ]) + (Y1,)

# frac=0.5
df_train2=df_train.sample(frac=0.5,random_state=1).reset_index()
df_train2.Y*=2
df_train2.X*=2
df_train2.rtt*=0.000001

noisertt2 = np.random.normal(0,1,len(df_train2)) #add noise
df_train2["rtt"]+=noisertt2

df_train2 = df_train2[[y_label, *condition_labels]]
df_train2["y_input"] = df_train2[y_label]
df_train2 = df_train2.drop(columns=[y_label])

batch_size2 = 5000
X2 = df_train2[condition_labels]
Y2 = df_train2.y_input
steps_per_epoch2 = len(df_train2) // batch_size2

X2 = np.array(X2)
Y2 = np.array(Y2)
training_data2 = tuple([ X2[:,i] for i in range(len(condition_labels)) ]) + (Y2,)

# frac=0.01
df_train3=df_train.sample(frac=0.01,random_state=1).reset_index()
df_train3.Y*=2
df_train3.X*=2
df_train3.rtt*=0.000001

noisertt3 = np.random.normal(0,1,len(df_train3)) #add noise
df_train3["rtt"]+=noisertt3

df_train3 = df_train3[[y_label, *condition_labels]]
df_train3["y_input"] = df_train3[y_label]
df_train3 = df_train3.drop(columns=[y_label])

batch_size3 = 100
X3 = df_train3[condition_labels]
Y3 = df_train3.y_input
steps_per_epoch3 = len(df_train3) // batch_size3

X3 = np.array(X3)
Y3 = np.array(Y3)
training_data3 = tuple([ X3[:,i] for i in range(len(condition_labels)) ]) + (Y3,)

# frac=0.1
df_train4=df_train.sample(frac=0.1,random_state=1).reset_index()
df_train4.Y*=2
df_train4.X*=2
df_train4.rtt*=0.000001

noisertt4 = np.random.normal(0,1,len(df_train4)) #add noise
df_train4["rtt"]+=noisertt4

df_train4 = df_train4[[y_label, *condition_labels]]
df_train4["y_input"] = df_train4[y_label]
df_train4 = df_train4.drop(columns=[y_label])

batch_size4 = 1000
X4 = df_train4[condition_labels]
Y4 = df_train4.y_input
steps_per_epoch4 = len(df_train4) // batch_size4

X4 = np.array(X4)
Y4 = np.array(Y4)
training_data4 = tuple([ X4[:,i] for i in range(len(condition_labels)) ]) + (Y4,)

# Validation
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
df_val.Y*=2
df_val.X*=2
df_val.rtt*=0.000001
df_val=np.array(df_val)
validation = df_val[(df_val[:,1]==4) & (df_val[:,2]==5)]

# Gaussian modal 
# 1
model1 = ConditionalGaussianMM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)
model1.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.05,
    ),
    loss=model1.loss,
)
model1.training_model.fit(
    x=training_data1,
    y=Y1,
    steps_per_epoch=steps_per_epoch1,
    epochs=3000,
    verbose=1,
)
#0.5
model2 = ConditionalGaussianMM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)
model2.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.05,
    ),
    loss=model2.loss,
)
model2.training_model.fit(
    x=training_data2,
    y=Y2,
    steps_per_epoch=steps_per_epoch2,
    epochs=3000,
    verbose=1,
)
#0.01
model3 = ConditionalGaussianMM(
    x_dim=condition_labels,
    centers=800,
    hidden_sizes=[8,8,8],
    dtype="float32",
)
model3.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.05,
    ),
    loss=model3.loss,
)
model3.training_model.fit(
    x=training_data3,
    y=Y3,
    steps_per_epoch=steps_per_epoch3,
    epochs=3000,
    verbose=1,
)
#0.1
model4 = ConditionalGaussianMM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)
model4.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.05,
    ),
    loss=model4.loss,
)
model4.training_model.fit(
    x=training_data4,
    y=Y4,
    steps_per_epoch=steps_per_epoch4,
    epochs=3000,
    verbose=1,
)


x1=np.array([[4,5],[4,5]])
y=np.arange(7,57,0.5)
p1=np.zeros(y.size)
p2=np.zeros(y.size)
p3=np.zeros(y.size)
p4=np.zeros(y.size)
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob, logprob, pred_cdf = model1.prob_batch(x1, yr)
    p1[i]=1-pred_cdf[0]
    p1[i+1]=1-pred_cdf[1]
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob2, logprob2, pred_cdf2 = model2.prob_batch(x1, yr)
    p2[i]=1-pred_cdf2[0]
    p2[i+1]=1-pred_cdf2[1]
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob3, logprob3, pred_cdf3 = model3.prob_batch(x1, yr)
    p3[i]=1-pred_cdf3[0]
    p3[i+1]=1-pred_cdf3[1]
for i in range(0,y.size-1,2):
    yr=np.array([y[i],y[i+1]])
    prob4, logprob4, pred_cdf4 = model4.prob_batch(x1, yr)
    p4[i]=1-pred_cdf4[0]
    p4[i+1]=1-pred_cdf4[1]
# Evaluation
ye=np.array([10])
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
d3=np.zeros(y.size)
d4=np.zeros(y.size)
d3=abs(p3-valp1) # substract
d4=abs(p4-valp1)
lp3=numpy.log10(p3+1e-10)
lp4=numpy.log10(p4+1e-10)
ld3=abs(lp3-lvalp1)
ld4=abs(lp4-lvalp1)

# Plot
# log
plt.figure(1)
plt.plot(y,valp1,'b--',label='DVP from dataset')
plt.plot(y,p1,'r--',label='DVP from Gaussian mixture model for frac=1')
plt.plot(y,p2,'g--',label='DVP from Gaussian mixture model for frac=0.5')
plt.plot(y,p4,'m--',label='DVP from Gaussian mixture model for frac=0.1')
plt.plot(y,p3,'y--',label='DVP from Gaussian mixture model for frac=0.01')
plt.legend()
plt.xlim([7,22])
plt.title('DVP comparison for validation point [4,5]')
plt.xlabel('target delay (ms)')
plt.ylabel('DVP')
plt.savefig("sdvp comparison.png") 

plt.figure(2)
plt.plot(y,valp1,'b--',label='DVP from dataset')
plt.plot(y,p1,'r--',label='DVP from Gaussian mixture model for frac=1')
plt.plot(y,p2,'g--',label='DVP from Gaussian mixture model for frac=0.5')
plt.plot(y,p4,'m--',label='DVP from Gaussian mixture model for frac=0.1')
plt.plot(y,p3,'y--',label='DVP from Gaussian mixture model for frac=0.01')
plt.yscale('log') # turn y axis to log
plt.legend()
plt.xlim([7,22])
plt.title('log DVP comparison for validation point [4,5]')
plt.xlabel('target delay (ms)')
plt.ylabel('DVP')
plt.savefig("slog dvp comparison.png")

plt.figure(3)
plt.plot(y,ld1,'r--',marker ='.',label='DVP from Gaussian mixture model for frac=1')
plt.plot(y,ld2,'g--',marker ='.',label='DVP from Gaussian mixture model for frac=0.5')
plt.plot(y,ld4,'m--',marker ='.',label='DVP from Gaussian mixture model for frac=0.1')
plt.plot(y,ld3,'y--',marker ='.',label='DVP from Gaussian mixture model for frac=0.01')
plt.legend()
plt.xlabel('target delay (ms)')
plt.ylabel('log DVP error')
plt.title('Log DVP error for validation point [4,5]')
plt.ylim([0,4])
plt.xlim([7,22])
plt.savefig("slog dvp error.png") 
