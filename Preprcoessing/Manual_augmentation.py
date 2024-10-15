### This function is used for right angle rotation of images in ImageDataGenerator() function.

def right_angle_rotate(input_image):
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        input_image = apply_affine_transform(
            input_image, theta=angle, fill_mode='nearest')
    return input_image
 
### Only right angle rotations plus horizontal and vertical flipping and change of brightness in the range of (0.8, 1.3)   
datagen = ImageDataGenerator(
        rotation_range=0,
        horizontal_flip=True, 
        brightness_range=(0.8, 1.3),
        vertical_flip=True,
        fill_mode='nearest',
        preprocessing_function=right_angle_rotate,
        validation_split=False)
        
        

       
