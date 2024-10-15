 # For random augmentation: rotation (-20 to 20)/flipping/shifting/brightness change (0.8 to 1.3)
 datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True, 
        fill_mode='nearest',
        brightness_range=(0.8, 1.3),validation_split= False)
       
