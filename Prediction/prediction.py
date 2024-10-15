import sys
import os
import json


# The directory of all packages my_imports.py
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../model/"))


# import all necessary packages
from my_imports import *
from model import model_definition.pretrained_model


# Load the parameters from the JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Access the parameters
excel_label                   = config['excel_label']
validation_slide              = config['validation_slide']
model_name                    = config['model_name']
validation_DATA_PATH          = config['validation_DATA_PATH']
validation_LABEL_PATH         = config['validation_LABEL_PATH']
fold_weights                  = config['fold_weights']
out_file                      = config['out_file']
input_shape                   = config['input_shape']
num_classes                   = config['num_classes']
activation_function           = config['activation_function']



### This is the labels of images and slides
df_scores = pd.read_excel(excel_label,sheet_name='Total Score',engine='openpyxl')

#######################################################################

### I took the best models in terms of TP 
validation_slide_weights = weights
tile_based_scores = {}
image_based_scores = {}
slide_based_scores = {}
for val_s in validation_slide:
    val_dir          = val_s # loop over all groups as validation dataset
   


    validation_DATA_PATH  =  val_dir+ validation_DATA_PATH 
    validation_LABEL_PATH =  val_dir+ validation_LABEL_PATH

    ### Max and min of Median column 
    max_score = max(df_scores['Median'])
    min_score = min(df_scores['Median'])

    ##Normalized thresholds
    low = (10-(min_score))/(max_score-min_score)  ##th1 = 10
    mid = (25-(min_score))/(max_score-min_score)  ##th2 = 25
    
    ## Initialize validation samples
    validation_images = glob.glob(validation_DATA_PATH+'*.tif')
    n_v= len(validation_images) 
    validation_samples = np.zeros((n_v, input_shape[0], input_shape[1], 3), dtype=np.float32)
    validation_label = np.zeros((n_v,), int)
    # Reading true labels of validation set


    precise_scores = []
    image_names = []
    with open(validation_LABEL_PATH,'r') as f:
        for ctr,line in zip(range(n_v),f):
            validation_samples[ctr,:,:,:] = cv2.imread(validation_DATA_PATH+(line.split()[0]))
            la = float(line.split()[1])
            
            if la > mid:
                validation_label[ctr] = 2
            elif la > low:
                validation_label[ctr] = 1
            else:
                validation_label[ctr] = 0
            precise_scores.append(validation_label[ctr])
            image_names.append(line.split(' ')[0 ])  
            
   
    for i in range(n_v):
        mod_img = validation_samples[i,:, :].astype(np.float32)
        validation_samples[i, :, :, :] = mod_img
    ##After we took all validation tiles we defined the model   
    




    ######## Loading models and weights #####################################################################
    pretrained_model = pretrained_model(input_shape, 3,'relu') ## 
    weights = fold_weights[val_dir]
    
    # Load weights, select between best_weights (val_accuracy) and final_weights from last epoch
    pretrained_model.load_weights(weights)

    # Make predictions on validation samples
    preds = pretrained_model.predict(validation_samples)
    
    # Max predictions in numpy array
    y_pred = np.argmax(preds,axis=1)  # 0,1 or 2
    x_l = range(len(y_pred))

    #########################################################################################################
    ### Here all predicted labels are saved in the following file
    with open(validation_LABEL_PATH,'r') as f: 
        with open(model_name+'_'+val_dir+ out_file,'w') as fout: 
            for i,line in zip(y_pred,f):
                print(line.split(' ')[0 ],np.int(i),sep=' ',file=fout)
        fout.close()
    f.close()

    #############################################################################################################
    ### Here we need to have tile, image and slide score to be able to compare with gold standard slide labels###
    ### This is for extracting slide number from image names to add
    keys = []
    Slide_dict = {}
    with open(model_name+'_'+val_dir+out_file,'r') as fout: 
        for line in fout:
            S_num = str(line.split(' ')[0][6:8])
            if '.' in S_num:
                S_num = S_num.replace('.','')
            if '_' in S_num:
                S_num = S_num.replace('_','')
            if S_num not in keys:
                keys.append(S_num)
                Slide_dict[S_num]=[]

    fout.close()
    ###############################################################################################################
    ### Add all scores of tiles to the slide key    e.g., '28' = [0,1,1,2,0,0,2,1]
    with open(model_name+'_'+val_dir + out_file,'r') as fout: 
        for line in fout:
            key_temp=line.split(' ')[0][6:8]
            if '.' in key_temp:
                key_temp = key_temp.replace('.','')
            if '_' in key_temp:
                key_temp = key_temp.replace('_','')
           
            Slide_dict[key_temp].append(float(line.split(' ')[1].strip()))

    #################################################################################################################
    #### Change all tile scores to the score of majority of tiles (majority vote)
    y_pred_slide = np.ones(len(y_pred))  
    offset = 0
    for k in Slide_dict.keys():
        class_0 = [index for index, element in enumerate(Slide_dict[k]) if element == 0]
        class_1 = [index for index, element in enumerate(Slide_dict[k]) if element == 1]
        class_2 = [index for index, element in enumerate(Slide_dict[k]) if element == 2]
        majority_vote = np.argmax([len(class_0),len(class_1),len(class_2)]) 
        y_pred_slide[offset:offset+len(Slide_dict[k])]= majority_vote
        offset += len(Slide_dict[k])
    #################################################################################################################





    
    #################################################################################################################
    ### Add all scores of tiles to the Image name key################################################################
    keys = []
    Image_dict = {}
    with open(model_name+'_'+val_dir + out_file,'r') as fout: 
        for line in fout:
            I_num = str(line.split('___')[0])
            if I_num not in keys:
                keys.append(I_num)
                Image_dict[I_num]=[]

    fout.close()
    
   
    with open(model_name+'_'+val_dir + out_file,'r') as fout: 
        for line in fout:
            key_temp=str(line.split('___')[0])
            Image_dict[key_temp].append(float(line.split(' ')[1].strip()))

    
    #################################################################################################################
    #### Change all tile scores to the score of majority of tiles for each image #################################### 
    y_pred_image = np.ones(len(y_pred))  
    offset = 0
    for k in Image_dict.keys():
        class_0 = [index for index, element in enumerate(Image_dict[k]) if element == 0]
        class_1 = [index for index, element in enumerate(Image_dict[k]) if element == 1]
        class_2 = [index for index, element in enumerate(Image_dict[k]) if element == 2]
        majority_vote = np.argmax([len(class_0),len(class_1),len(class_2)]) 
        y_pred_image[offset:offset+len(Image_dict[k])]= majority_vote
        offset += len(Image_dict[k])

    
    #################################################################################################################
    #############################This part is good to run on jupyter notebooks or save them in a figure
    
    plt.figure()
    plot_confusion_matrix(precise_scores, y_pred)  #Tile based score
    plt.title('Tile based Confusion matrix')
    plt.savefig('Tile_CM.png')  # Save as png
    
    plt.figure()
    plot_confusion_matrix(precise_scores, y_pred_image)  ## Image based score 
    plt.title('Image based Confusion matrix')
    plt.savefig('Image_CM.png')  # Save as png
    
    plt.figure()
    plot_confusion_matrix(precise_scores, y_pred_slide)  ## Slide based score
    plt.title('Slide based Confusion matrix')
    plt.savefig('Slide_CM.png')  # Save as png

    ##plt.show()

    


    #### This also gives nice report of this comparison############################################
    print(classification_report(precise_scores,y_pred))
    print(classification_report(precise_scores,y_pred_image))
    print(classification_report(precise_scores,y_pred_slide))
    
    ##############################################################################################
    tile_based_scores[val_dir]  = pd.DataFrame(classification_report(precise_scores,y_pred,output_dict=True)).transpose()
    image_based_scores[val_dir] = pd.DataFrame(classification_report(precise_scores,y_pred_image,output_dict=True)).transpose()
    slide_based_scores[val_dir] = pd.DataFrame(classification_report(precise_scores,y_pred_slide,output_dict=True)).transpose()
    




df_tile  = pd.concat([tile_based_scores['1st_part'], tile_based_scores['2nd_part'],tile_based_scores['3rd_part']], axis=1,keys=['fold1','fold2','fold3'])
df_image = pd.concat([image_based_scores['1st_part'], image_based_scores['2nd_part'],image_based_scores['3rd_part']], axis=1,keys=['fold1','fold2','fold3'])
df_slide = pd.concat([slide_based_scores['1st_part'], slide_based_scores['2nd_part'],slide_based_scores['3rd_part']], axis=1,keys=['fold1','fold2','fold3'])


df_tile.to_csv(model_name+'_tiles.csv')
df_image.to_csv(model_name+'_image.csv')
df_slide.to_csv(model_name+'_slide.csv')