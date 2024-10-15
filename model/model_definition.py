import sys
import os

# The directory of all packages my_imports.py
sys.path.append(os.path.abspath("../"))

# import all necessary packages
from my_imports import *



def pretrained_model(img_shape, num_classes,layer_type):
    model_vgg16_conv = EfficientNetB4(weights='imagenet', include_top=False)  ##  or other models

    #Input format
    keras_input = Input(shape=(224,224,3), name = 'image_input')
    
    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(keras_input)
    
    for layer in model_vgg16_conv.layers:
        layer.trainable = False
    
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(128, activation=layer_type, name='fc1')(x)
    x = Dense(128, activation=layer_type, name='fc2')(x)
    x = Dropout(0.5, name='dropout')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    METRICS = [
      keras.metrics.CategoricalAccuracy(name='accuracy')]
    
    
    #Create your own model 
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=METRICS)
    pretrained_model.summary()

    return pretrained_model
