#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:30:44 2019

@author: Andrew Robbertz
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, LeakyReLU, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

# Set File Save Locations
g_fig = '.\images\models\gen-conv_lg_30dr_500e.png'
d_fig = '.\images\models\disc-conv_lg_30dr_500e.png'
GAN_fig = '.\images\models\GAN-conv_lg_30dr_500e.png'
checkpoint_loc = '.\checkpoints\_conv_lg_30dr_500e\checkpoint'
checkpoint_fig = '.\images\_renders\_conv_lg_30dr_500e\checkpoint'

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f'Pre-Expansion X_train.shape : {X_train.shape}')

# Preprocessing
# X_train = X_train.reshape(-1, 784)
X_train = np.expand_dims(X_train, axis=-1)
X_train = X_train.astype('float32')/255

print(f'Post-Expansion X_train.shape : {X_train.shape}')

# Set the dimensions of the noise
z_dim = 100

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)


# Generator
g = Sequential()
# foundation for 7x7 image
n_nodes = 128 * 7 * 7
g.add(Dense(n_nodes, input_dim=z_dim))
g.add(LeakyReLU(alpha=0.2))
g.add(Reshape((7, 7, 128)))
# upsample to 14x14
g.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
g.add(LeakyReLU(alpha=0.2))
# upsample to 28x28
g.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
g.add(LeakyReLU(alpha=0.2))
g.add(Conv2D(1, (7,7), activation='sigmoid', padding='same')) 
plot_model(g, to_file=g_fig, show_shapes=True)

# Discrinimator
in_shape = (28, 28, 1)
d = Sequential()
d.add(Conv2D(128, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
d.add(LeakyReLU(alpha=0.2))
d.add(Dropout(0.3))
d.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
d.add(LeakyReLU(alpha=0.2))
d.add(Dropout(0.3))
d.add(Flatten())
d.add(Dense(1, activation='sigmoid'))
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
plot_model(d, to_file=d_fig, show_shapes=True)

# GAN
d.trainable = False
inputs = Input(shape=(z_dim, ))
hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
plot_model(gan, to_file=GAN_fig, show_shapes=True)

# summarize layers
print(gan.summary())

# Training
def train(epochs=1, plt_frq=1, BATCH_SIZE=128):
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)

    history = {
        'val_acc': [],
        'val_loss': [],
        'g_acc': [],
        'g_loss': [],
        'd_acc': [],
        'd_loss': [],
    }
    
    for e in (range(1, epochs+1)):
        localhist = {
            'g_acc': [],
            'g_loss': [],
            'd_acc': [],
            'd_loss': [],
        }

        for batch_no in range(batchCount):  
            # Create a batch by drawing random index numbers from the training set
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
            # Create noise vectors for the generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            
            # Generate the images from the noise
            generated_images = g.predict(noise)
            X = np.concatenate((image_batch, generated_images))
            # Create labels
            y = np.zeros(2*BATCH_SIZE)
            y[:BATCH_SIZE] = 1
    
            # Train discriminator on generated images
            d.trainable = True
            d_loss, d_acc = d.train_on_batch(X, y)
    
            # Train generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            y2 = np.ones(BATCH_SIZE)
            d.trainable = False
            g_loss, g_acc = gan.train_on_batch(noise, y2)

            # Save Hostory 
            localhist['d_acc'].append(d_acc)
            localhist['d_loss'].append(d_loss)
            localhist['g_acc'].append(g_acc)
            localhist['g_loss'].append(g_loss)
    

        # Validation
        noise_val = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
        y_val = np.ones(BATCH_SIZE)
        val_loss, val_acc = gan.test_on_batch(noise_val, y_val)
        # Save History
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['d_acc'].append(sum(localhist['d_acc'])/len(localhist['d_acc']))
        history['d_loss'].append(sum(localhist['d_loss'])/len(localhist['d_loss']))
        history['g_acc'].append(sum(localhist['g_acc'])/len(localhist['g_acc']))
        history['g_loss'].append(sum(localhist['g_loss'])/len(localhist['g_loss']))

        # Print Epoch Stats
        print(f'Epoch {e:3d} || Discriminator Accuracy : {d_acc:.5f} Loss : {d_loss:3.5f}  || Generator Accuracy : {g_acc:.5f} Loss : {g_loss:3.5f} || Validation Accuracy: {val_acc:.5f} Loss : {val_loss:3.5f}')

        # Checkpoint
        if e % 10 == 0:
            print(f'Saving Model Checkpoint in Epoch : {e}')
            save_model(model=g, f_name=checkpoint_loc, checkpoint_no=e)
            generate_images(model=g, f_name=checkpoint_fig, checkpoint_no=e)

    return history

def save_model(model, f_name='generator', checkpoint_no=0):
    # serialize model to JSON
    model_json = g.to_json()
    with open(f'{f_name}_e{checkpoint_no}.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    g.save_weights(f'{f_name}_e{checkpoint_no}.h5')
    # print("Saved model to disk")

def generate_images(model, f_name='generator', checkpoint_no=0):
    # Generate images
    np.random.seed(586)
    h = w = 28
    num_gen = 25

    z = np.random.normal(size=[num_gen, z_dim])
    generated_images = g.predict(z)

    # plot of generation
    n = np.sqrt(num_gen).astype(np.int32)
    I_generated = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(I_generated, cmap='gray')
    plt.savefig(f'{f_name}_e{checkpoint_no}.png')


# Do Some Training 
history = train(epochs=500, plt_frq=1, BATCH_SIZE=256)

# Save the Model
save_model(model=g, f_name=checkpoint_loc, checkpoint_no='_final')
# Generate images
generate_images(model=g, f_name=checkpoint_fig, checkpoint_no='_final')
 
# Plot Accuracy and Loss
plt.subplot(1, 2, 1)
plt.plot(history['d_acc'], label='Discriminator Accuracy')
plt.plot(history['g_acc'], label='Generator Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history['d_loss'], label='Discriminator Loss')
plt.plot(history['g_loss'], label='Generator Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()