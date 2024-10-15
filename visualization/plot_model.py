
import sys
import os

# The directory of all packages my_imports.py and model
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../model/"))

# import all necessary packages
from my_imports import *


from my_imports import * 
from model_definition import pretrained_model


## Here model will be plotted and saved in *.png file
model = pretrained_model([224,224,3],3,'relu')
plot_model(model, to_file="effnetb4_final.png", show_shapes=True)