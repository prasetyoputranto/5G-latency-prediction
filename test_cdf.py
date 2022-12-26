import numpy
import pandas as pd
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
df_train = df_train[[y_label, *condition_labels]]
df_train["y_input"] = df_train[y_label]
df_train = df_train.drop(columns=[y_label])

batch_size = 1000
X = df_train[condition_labels]
Y = df_train.y_input
steps_per_epoch = len(df_train) // batch_size

X = np.array(X)
Y = np.array(Y)
training_data = tuple([ X[:,i] for i in range(len(condition_labels)) ]) + (Y,)

# # Gaussian modal
# model = ConditionalGaussianMM(
#     x_dim=condition_labels,
#     centers=3,
#     hidden_sizes=[8,8,8],
#     dtype="float32",
# )

# model.training_model.compile(
#     optimizer=tf.keras.optimizers.Adam(
#         learning_rate=0.001,
#     ),
#     loss=model.loss,
# )

# model.training_model.fit(
#     x=training_data,
#     y=Y,
#     steps_per_epoch=steps_per_epoch,
#     epochs=1500,
#     verbose=1,
# )
# x1=np.array([[1,5],[1,5]])
# y=np.arange(250)
# p1=np.zeros(y.size)
# for i in range(0,y.size-1,2):
#     yr=np.array([y[i],y[i+1]])
#     prob, logprob, pred_cdf = model.prob_batch(x1, yr)
#     p1[i]=prob[0]
#     p1[i+1]=prob[1]
#Try another condition
# x2=np.array([[10],[10]])
# p2=np.zeros(y.size)
# for i in range(0,y.size-1,2):
#     yr2=np.array([y[i],y[i+1]])
#     prob, logprob, pred_cdf = model.prob_batch(x2, yr2)
#     p2[i]=prob[0]
#     p2[i+1]=prob[1]



# Gamma modal
model = ConditionalGammaMixtureEVM(
    x_dim=condition_labels,
    centers=3,
    hidden_sizes=[8,8,8],
    dtype="float32",
)

model.training_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
    ),
    loss=model.loss,
)
model.training_model.fit(
    x=training_data,
    y=Y,
    steps_per_epoch=steps_per_epoch,
    epochs=300,
    verbose=1,
)


# Get cdf
y_Gamma=np.array([70])
x0=np.arange(0,20)
x1=np.arange(0,20)
X0,X1=np.meshgrid(x0,x1)

c_Gamma=np.zeros((x0.size,x1.size))
for i in range(0,x0.size-1):
    for j in range(0,x1.size-1):
        prob, logprob, pred_cdf = model.prob_batch(np.array([[X0[i][j],X1[i][j]]]), y_Gamma)
        c_Gamma[i][j]=pred_cdf

# Plot
ax = plt.axes(projection='3d')
ax.plot_surface(X0,X1,c_Gamma,alpha=0.3,cmap='winter')
# ax.contour(X0,X1,c_Gamma,zdir='z', offset=-3,cmap="rainbow")
ax.contour(X0,X1,c_Gamma,zdir='x', offset=-3,cmap="rainbow")
ax.contour(X0,X1,c_Gamma,zdir='y', offset=-3,cmap="rainbow")
ax.view_init(elev=45, azim=45)
ax.set_xlabel('queue0')
ax.set_ylabel('queue1')
ax.set_zlabel('cdf')
ax.set_xlim(-2, 20)
ax.set_ylim(-2, 20)
ax.set_zlim(-0.5, 1)
