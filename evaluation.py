import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model,model_from_json
import numpy as np


def load_model(json_model="generator.json", h5_file="generator.h5"):
    '''
    This function is used to load model, codes below are based on template.py.
    Please modify this function based on your own codes.
    '''
    with open(json_model, "r") as json_file:
        md_json = json_file.read()
    t = model_from_json(md_json)
    t.load_weights(h5_file)
    return t

def generate_image(model):
    '''
    Take the model as input and generate one image, codes below are based on template.py.
    Please modify this function based on your own codes.
    '''
    # Set the dimensions of the noise
    z_dim = 100
    z = np.random.normal(size=[1, z_dim])
    generated_images = g.predict(z)
    return generated_images

def plot_image(model, num_images=25):
    # Dims of Each picture
    h = w = 28
    # Set the dimensions of the noise
    z_dim = 100
    z = np.random.normal(size=[num_images, z_dim])
    generated_images = model.predict(z)
    
    # plot of generation
    n = np.sqrt(num_images).astype(np.int32)
    I_generated = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(I_generated, cmap='gray')
    plt.show()


if __name__ == "__main__":
    json_model = '.\checkpoints\_ff_dr0_LR\checkpoint_e_final.json'
    h5_file = '.\checkpoints\_ff_dr0_LR\checkpoint_e_final.h5'

    model = load_model(json_model=json_model, h5_file=h5_file)
    # image = generate_image(model)
    plot_image(model, num_images=25)

