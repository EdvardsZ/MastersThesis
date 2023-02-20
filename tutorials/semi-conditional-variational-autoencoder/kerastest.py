from tensorflow import keras
import numpy as np


def Conditioning(data, obs_x, obs_y, fill_value=0.000001):
    Cond = np.zeros((data.shape))
    Cond[:,obs_x,obs_y]=np.copy(data[:,obs_x, obs_y])
    return Cond

#%%
##Select observations (Observations fixed during training)
start=2
stop=26
obs_x_n=6
obs_y_n=6

obs_x=[]
obs_y=[]
for i in range(start,stop,obs_x_n):
    for j in range(start,stop,obs_y_n):
        obs_x.append(i)
        obs_y.append(j)
print(obs_x)
print(obs_y)
## Obtain the MNIST dataset
latent_dim = 2
image_size = 28
input_shape = (image_size, image_size, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

## Extract the observations to condition on
cond_x_train = Conditioning(x_train, obs_x, obs_y)
Cond_X_train=np.expand_dims(cond_x_train, axis=-1)
cond_x_test = Conditioning(x_test, obs_x, obs_y)
Cond_X_test=np.expand_dims(cond_x_test, axis=-1)

print(cond_x_train.shape)
print(Cond_X_train.shape)
print(Cond_X_test.shape)