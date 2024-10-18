### Two random operation will be performed on images such as flipping and changing the brightness of images in first function 
### Selecting the training batches is also random
### Blurring is performed in second part by filter size of 5x5

import numpy as np
import cv2
import glob
from random import random
from pathlib import Path

class ImageAugmentor:
    def __init__(self, brightness_range=(0.8, 1.2), blur_kernel_size=5):
        self.brightness_range = brightness_range
        self.blur_kernel_size = blur_kernel_size
    
    def adjust_brightness(self, im):
        """Randomly adjust brightness by a factor in the given range."""
        brightness_factor = self.brightness_range[0] + (self.brightness_range[1] - self.brightness_range[0]) * random()
        im = im * brightness_factor
        return np.clip(im, 0, 255)  # Clip the values to be within the valid range
    
    def flip_lr(self, im):
        """Randomly flip the image left-to-right with a 50% chance."""
        if random() > 0.5:
            im = np.fliplr(im)
        return im
    
    def blur_image(self, im):
        """Apply a blur with a kernel size defined in the class."""
        kernel = np.ones((self.blur_kernel_size, self.blur_kernel_size), np.float32) / (self.blur_kernel_size ** 2)
        return cv2.filter2D(im, -1, kernel)
    
    def random_augment(self, im):
        """Perform random brightness adjustment and random flipping."""
        im = self.adjust_brightness(im)
        im = self.flip_lr(im)
        im = im.astype(np.float32) / 255 - 0.5  # Normalize image as per original code
        return im
    
    def process_image(self, im, apply_blur=True):
        """Apply random augmentations and optional blurring."""
        x = random()
        if x > 0.5:
            im = self.random_augment(im)
        
        # Optionally apply blur to the image
        if apply_blur and random() > 0.5:
            im = self.blur_image(im)
        
        return im
    
    def flow(self, x, y, batch_size):
        """
        Generate batches of augmented images and labels.
        :param x: Training images.
        :param y: Corresponding labels.
        :param batch_size: Batch size.
        :return: A generator yielding batches of augmented images and their labels.
        """
        num_samples = x.shape[0]
        indices = np.arange(num_samples)
        while True:
            # Shuffle the indices for each epoch
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]

                # Augment each image in the batch
                augmented_images = np.array([self.process_image(img,apply_blur=True) for img in batch_x])

                yield augmented_images, batch_y  # Yield augmented images and their labels

