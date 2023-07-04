from glob import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.applications import VGG16


class VggRealizer:
    
    def __init__(self):
        self.vgg = VGG16(include_top=False, weights='imagenet')
        self.real_images_paths = list(glob('/home/klima7/studies/gsn/minecraft/Mc2Real-Downscale-Upscale/data/real_images/*.png'))
    
    def __call__(self, mc_image, num_steps=300):
        image = tf.Variable(
            initial_value=mc_image[None, ...],
            trainable=True,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        iterator = tqdm(range(num_steps))
        for step in iterator:
            with tf.GradientTape() as tape:
                real_image = self.__get_random_real_image()
                vgg_loss = self.__vgg_loss(image, real_image[None, ...])
                l1_loss = tf.math.reduce_sum(tf.math.abs(real_image[None, ...].astype(np.float32) - mc_image.astype(np.float32)))
                loss_value = 0.5 * vgg_loss + 0.5 * l1_loss
                
            gradients = tape.gradient(loss_value, [image])
            optimizer.apply_gradients(zip(gradients, [image]))
            
            iterator.set_postfix_str(f'Loss: {loss_value.numpy()}')
            
        return image.numpy()
    
    def __get_random_real_image(self):
        path = random.choice(self.real_images_paths)
        image = cv2.imread(str(path))
        image = cv2.resize(image, (64, 64))
        image = image / 255 - 0.5
        image = image[..., ::-1]
        return image
        
    @staticmethod
    def __vgg_loss(y_true, y_pred):
        vgg = VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        loss_model = tf.keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block4_conv3').output)
        true_features = loss_model(y_true)
        pred_features = loss_model(y_pred)
        mse_loss = MeanSquaredError()(true_features, pred_features)
        return mse_loss
