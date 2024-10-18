import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import apply_affine_transform

class RandomImageAugmentor:
    def __init__(self):
        # Initialize the ImageDataGenerator with the given configurations
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=(0.8, 1.3),
            validation_split=False
        )

    def augment(self, input_image):
        """
        Apply random transformations to the input image.
        :param input_image: Input image to augment.
        :return: Augmented image.
        """
        augmented_image = self.datagen.random_transform(input_image)
        return augmented_image

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
                augmented_images = np.array([self.augment(img) for img in batch_x])

                yield augmented_images, batch_y  # Yield augmented images and their labels