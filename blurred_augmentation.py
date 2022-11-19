### Two random operation will be performed on images such as flipping and changing the brightness of images in first function 
### Selecting the training batches is also random
### Blurring is performed in second part by filter size of 5x5

import numpy as np
import imageio as iio
from pathlib import Path
import cv2
from random import choice, random
import glob

main_path = PATH_to_Original_Images
dest_path = PATH_to_brightened_flipped_images

images = glob.glob(main_path + "/*.tif")
       
  
###############################################
def brightness_fliplr_random_process(im):
    brightness = 0.8 + 0.4*random()
    im *= brightness
    im = np.clip(im, 0, 255) # It would clip large values
    if random() > 0.5:
        # Flip right-left
        im = np.fliplr(im)
    new_img = np.clip(im, 0, 255)    
    new_img /= 255
    new_img -= 0.5    
    return new_img
##############################################

for i in images:
    name = i.split('/')[-1]
    im = cv2.imread(i)
    
    x = random()
    if x > 0.5:
        mod_img = im.astype(np.float32)
        new_img =  brightness_fliplr_random_process(mod_img)
    else:
        new_img = im

    
    cv2.imwrite(dest_path+'/'+name, new_img)
    
##############################################
main_path = PATH_to_brightened_flipped_images
dest_path = Path_to_blurred_Imagez

images = glob.glob(main_path + "/*.tif")
       
    
k_size = 5 ##5, 10, 25, 40
kernel = np.ones((k_size,k_size),np.float32)/(k_size**2)    
##############################################
for i in images:
    name = i.split('/')[-1]  ## remove the path from name of the image
    im = cv2.imread(i)
    x = random()
    if x > 0.5:
        new_img = cv2.filter2D(im,-1,kernel)
    else:
        new_img = im

    
    cv2.imwrite(dest_path+'/'+name, new_img)
    
    
    
    

    
