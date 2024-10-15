#packages
from tensorflow.python.keras.models import Sequential,Model
from keras.applications.vgg16 import VGG16

from tensorflow.python.keras.layers import Reshape,GlobalAveragePooling2D,DepthwiseConv2D,Multiply,Add
from collections import defaultdict
import visualkeras
from PIL import ImageFont
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB4,EfficientNetB7
import keras
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
)



## Custom function for calculating TP for multi-class classification for each epoch
class MulticlassTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='multiclass_true_positives', **kwargs):
        super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)

### define model parameters
layer_type='relu'
num_classes = 3

conv_base= VGG16(input_shape=(224,224,3),weights='imagenet', include_top=False)  ##It could be Effnet or any other model
# Create a new 'top' of the model (i.e. fully-connected layers). 
top_model = conv_base.output
top_model = Flatten(name="flatten")(top_model)
top_model = Dense(128, activation='relu')(top_model)
top_model = Dense(128, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(3, activation='softmax')(top_model)

new_model = keras.models.load_model(
    weights, custom_objects={"MulticlassTruePositives": MulticlassTruePositives})
new_model.summary()

#########################################################################################
## Here we define color map for visualizing different layers
color_map = defaultdict(dict)
print(color_map)


color_map[Conv2D]['fill'] = '#104E8B'
color_map[ZeroPadding2D]['fill'] = '#7A378B'
color_map[BatchNormalization]['fill'] = '#E9967A'
color_map[Dropout]['fill'] = 'blue'
color_map[MaxPooling2D]['fill'] = 'pink'
color_map[Concatenate]['fill'] = '#20B2AA'
color_map[Flatten]['fill'] = 'teal'
color_map[Activation]['fill'] = 'red'
color_map[Add]['fill'] = 'cyan'
color_map[Multiply]['fill'] = 'navyblue'
color_map[Reshape]['fill'] = '#00FFFF'
color_map[GlobalAveragePooling2D]['fill'] = 'deeppink'
color_map[DepthwiseConv2D]['fill'] = '#8A2BE2'keys(),color_map.values())

font = ImageFont.truetype("Anaconda3/pkgs/matplotlib-base-3.3.2-py38hba9282a_0/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif.ttf", 200)

visualkeras.layered_view(conv_base, to_file='VGG16_toplayers_legend_new.png',type_ignore=[visualkeras.SpacingDummyLayer],
                         legend=True,draw_volume=True,scale_xy=100,spacing=0, font=font, color_map=color_map).show()
