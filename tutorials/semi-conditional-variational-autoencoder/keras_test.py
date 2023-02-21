from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse
from keras import backend as K

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

## Had to add this to make it work with the new version of tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

#Set cmap to viridis
plt.set_cmap('viridis')

#clear the graph 
K.clear_session()
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Implements reparameterization trick by sampling
    from a gaussian with zero mean and std=1.
    Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    Returns:
        sampled latent vector (tensor)
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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
(x_train, y_train), (x_test, y_test) = mnist.load_data()

## Extract the observations to condition on
Cond_X_train=np.expand_dims(Conditioning(x_train, obs_x, obs_y), axis=-1)
Cond_X_test=np.expand_dims(Conditioning(x_test, obs_x, obs_y), axis=-1)

## Preprosessing steps
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size,1])
x_test = np.reshape(x_test, [-1, image_size, image_size,1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

cond_x_train = Cond_X_train.astype('float32') / 255
cond_x_test = Cond_X_test.astype('float32') / 255

# compute the number of labels
num_labels = len(np.unique(y_train))

# network parameters
input_shape = (image_size, image_size,1)
label_shape = (num_labels, )
batch_size = 64
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 60
plt.imshow(cond_x_train[1,:,:,0])

#%%

#SCVAE model - encoder
def GundNet(cond_x_train, kernel_size=2, filters=32, image_size=image_size):

    cond_shape = (cond_x_train.shape[1],cond_x_train.shape[2],1)
    
    inputs = Input(shape=input_shape, name='encoder_input')
    
    x = Reshape((image_size, image_size, 1))(inputs)
    
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
    
    x = Conv2D(filters=filters*2,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
    
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model([inputs, cond_input], [z_mean, z_log_var, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    cond_input = Input(shape=cond_shape, name='conditional_input')

    x = Flatten()(cond_input)
    x = keras.layers.concatenate([latent_inputs, x])
    
    x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    
    x = Conv2DTranspose(filters=filters*2,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

    x = Conv2DTranspose(filters=filters,
                       kernel_size=kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

    
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='linear',
                              padding='same',
                              name='decoder_output')(x)
    
    #Instantiate decoder model
    decoder = Model([latent_inputs, cond_input], outputs, name='decoder')
 
    #Instantiate vae model
    outputs = decoder([encoder([inputs, cond_input])[2], cond_input])
    scvae = Model([inputs, cond_input], outputs, name='cvae')
    
    #The weigth has to be defined. They are updated through training. 
    weight_kl = tf.keras.backend.variable(0.) 
    weight_recon = tf.keras.backend.variable(1.)
    
    #Loss to be trained for the SCVAE. 
    def scvae_loss(weight_kl, weight_recon):
        def loss(y_true, y_pred):
            # mse lossmse(K.flatten(Train_inputs), K.flatten(outputs))
            reconstruction_loss = mse(K.flatten(y_true[0]), K.flatten(y_pred[0]))
            reconstruction_loss *= 784 
            
            # kl loss
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5

            out_loss = K.mean(weight_recon*reconstruction_loss + weight_kl*kl_loss)
            return out_loss
        return loss
    
    #Monitoring the kl-loss and reconstruction loss seperatly
    def kl_loss(y_true, y_pred):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(kl_loss)
    
    def recon_loss(y_true, y_pred):
        reconstruction_loss1 = mse(K.flatten(y_true[0]), K.flatten(y_pred[0]))
        reconstruction_loss1 *= 784
        return reconstruction_loss1
    
    #The weights of the kl-loss and reconstruction loss is used to adpativley adjust 
    #the weight during training. See also Adaptive Callback.
    def kl_out_weight(y_true, y_pred):
        return weight_kl

    def recon_out_weight(y_true, y_pred):
        return weight_recon

    #The model is compiled, and the kl-loss, reconstruction loss, the weights of the recon and
    #kl loss is tracked. Tracking and updating of the weights are done through a callback.  
    scvae.compile(optimizer='adam', 
                  metrics=[kl_loss, recon_loss, kl_out_weight, recon_out_weight],
                  loss=[scvae_loss(weight_kl, weight_recon)])

    return encoder, decoder, scvae, weight_kl, weight_recon    
    
#%%
encoder, decoder, scvae, weight_kl, weight_recon = GundNet(cond_x_train, kernel_size=2, filters=128)
encoder.summary()
decoder.summary()
scvae.summary()


#%%
class Adaptive_Callback(Callback):
    """Callback that terminates training when either acc or val_acc reaches
    a specified baseline
    """
    def __init__(self, weight_kl, weight_recon, monitor_kl='val_kl_loss', monitor_recon='val_recon_loss', start_epoch=0):
        super(Adaptive_Callback, self).__init__()
        self.monitor_kl = monitor_kl
        self.monitor_recon = monitor_recon
        self.weight_kl = weight_kl
        self.weight_recon = weight_recon
        self.start_epoch = start_epoch
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.kl_loss = logs.get(self.monitor_kl)
        self.recon_loss = logs.get(self.monitor_recon)
        #print("Current KL Weight is " + str(tf.keras.backend.get_value(self.weight_kl)))
        
        if self.start_epoch >= epoch:
            new_weight_kl = 0 #self.kl_loss/(self.kl_loss+self.recon_loss)
            new_weight_recon = 1 #self.recon_loss/(self.kl_loss+self.recon_loss)
        else:
            new_weight_kl = self.kl_loss/(self.kl_loss+self.recon_loss)
            new_weight_recon = self.recon_loss/(self.kl_loss+self.recon_loss)
        K.set_value(self.weight_kl, new_weight_kl)
        K.set_value(self.weight_recon, new_weight_recon)
        print("\n Current KL Weight is " + str(K.get_value(self.weight_kl)))
        print ("Current Recon Weight is " + str(K.get_value(self.weight_recon)))

#%%
#Callbacks - Early stopping and Annealing Callback
early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min', restore_best_weights=True)
adapt_callback = Adaptive_Callback(weight_kl, weight_recon)

#fit Scvae
scvae = scvae.fit([x_train, cond_x_train], [x_train],
                 epochs=100,
                 batch_size=128,
                 verbose=1,
                 callbacks=[adapt_callback, early_stop],
                 validation_split=0.2)


#%%
## Plot loss of training 
plt.figure(figsize=(16, 6))
plt.subplot(1,3,1)

#Plot training/validation loss 
plt.plot(scvae.history['loss'])
plt.plot(scvae.history['val_loss'])
plt.ylabel('kl loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss'], loc='upper right')

#plot training/validation kl loss
plt.subplot(1,3,2)
plt.plot(scvae.history['kl_loss'])
plt.plot(scvae.history['val_kl_loss'])
plt.ylabel('kl loss')
plt.xlabel('epoch')
plt.legend(['train kl-loss', 'test kl-loss'], loc='upper right')


#plot training/validataion recon loss
plt.subplot(1,3,3)
plt.plot(scvae.history['recon_loss'])
plt.plot(scvae.history['val_recon_loss'])
plt.ylabel('Recon loss')
plt.xlabel('epoch')
plt.legend(['train recon loss', 'test recon loss'], loc='upper right')

plt.show()


#%%

# display a 2D plot of the digit classes in the latent space

#First predic based on the data
z_mean, _, z = encoder.predict([x_test, cond_x_test],
                               batch_size=32)

#plot latent space of the test data set
plt.figure(figsize=(8, 6))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

#%%
#filename = os.path.join(model_name, "%05d.png" % np.argmax(y_label))
# display a 10x10 2D manifold of the digit (y_label)
n = 10
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
sample=[507]
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-3, 3, n)
grid_y = np.linspace(-3, 3, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict([z_sample, cond_x_test[sample,::]])
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(8, 8))
start_range = digit_size // 2
end_range = n * digit_size + start_range + 1
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
#plt.xticks(pixel_range, sample_range_x)
#plt.yticks(pixel_range, sample_range_y)
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.imshow(figure, cmap='Greys_r')
#plt.savefig(filename)
plt.show()
    

#%%
#sample=100
def plot_var(encoder, decoder, cond_x_test, sample):
    n = 6
    digit_size = 28
    sample=sample
    
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]
    
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict([z_sample, cond_x_test[[sample],:,:]])
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1,3,3)
    
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    #plt.xticks(pixel_range, sample_range_x)
    #plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title('Prediction with uniform sampling in the latent space')
    #plt.imshow(figure, cmap='Greys_r')
    im=ax.imshow(figure)
    #y_label
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cb = fig.colorbar(im, cax=cax)
    
    ax1 = fig.add_subplot(1,3,1)
    
    im2=ax1.imshow(x_test[sample,:,:].reshape(digit_size, digit_size))
    plt.title('True label - {}'.format(y_test[sample]))
    #divider = make_axes_locatable(ax1)
    #cax1 = divider.append_axes("right", size="5%", pad=0.05)
    #cb1 = fig.colorbar(im, cax=cax1)
    #ax1.get_yaxis().set_ticks([])
    
    ax2 = fig.add_subplot(1,3,2)
    plt.title('Subsampling from the true label')
    im3=ax2.imshow(cond_x_test[sample,:,:,:].reshape(digit_size, digit_size))
    plt.tight_layout()
    #plt.show()
    return im, im2, im3, sample_range_x, sample_range_y, pixel_range, figure

#%%
fig=plot_var(encoder, decoder, cond_x_test, sample=507)

#%%
#functions for assessing the error of the reconstruction...

def error(x_test, pred):
    error = np.sum(np.square(np.ndarray.flatten(pred) - np.ndarray.flatten(x_test)))/np.sum(np.square(np.ndarray.flatten(x_test))) 
    return error

def error_2(x_test, pred):
    mean_error=[]
    for i in range(0, x_test.shape[0]):
        mean_error.append(np.sum(np.square(np.ndarray.flatten(pred[i,...]) - np.ndarray.flatten(x_test[i,...])))/np.sum(np.square(np.ndarray.flatten(x_test[i,...]))))
    return np.array(mean_error) 


