### This function is used for right angle rotation of images in ImageDataGenerator() function.
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import apply_affine_transform

        
 class ImageAugmentor:
    def __init__(self):
        # Initialize ImageDataGenerator with your configurations
        self.datagen = ImageDataGenerator(
            rotation_range=0,
            horizontal_flip=True,
            brightness_range=(0.8, 1.3),
            vertical_flip=True,
            fill_mode='nearest',
            preprocessing_function=self.right_angle_rotate,
            validation_split=False
        )

    def right_angle_rotate(self, input_image):
        # Randomly choose an angle from 0, 90, 180, or 270 degrees
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            input_image = apply_affine_transform(
                input_image, theta=angle, fill_mode='nearest'
            )
        return input_image

    def augment(self, input_image):
        # Apply the data generator transformations to the input image
        input_image = self.datagen.random_transform(input_image)
        return input_image

    def flow(self, x, y, batch_size):
        # Create a generator for the training data
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
       

       
