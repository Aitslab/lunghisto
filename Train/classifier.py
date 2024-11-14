import sys
import os
import json


# The directory of all packages my_imports.py
sys.path.append(os.path.abspath("../"))
sys.path.append(os.path.abspath("../model/"))
sys.path.append(os.path.abspath("../Preprocessing/"))


# import all necessary packages
from my_imports import *
from model import model_definition.pretrained_model
from Preprocessing import Manual_augmentation.ImageAugmentor  ##Different augmentation
from Preprocessing import Random_augmentation.ImageAugmentor
from Preprocessing import brightness_blurred_augmentation.ImageAugmentor


## functions and objects required for trackcing results in each epoch (callbacks)
##########################augmentation############################
##################################################################
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

##################################################################
class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]             
        y_pred_class = np.argmax(y_pred, axis=1)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true, y_pred_class,labels=[0,1],true_labels=[0,1], ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))




# Load the parameters from the JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Access the parameters
excel_label                   = config['excel_label']
validation_slide              = config['validation_slide']
model_name                    = config['model_name']
train_DATA_PATH               = config['train_DATA_PATH']
train_LABEL_PATH              = config['train_LABEL_PATH']
validation_DATA_PATH          = config['validation_DATA_PATH']
validation_LABEL_PATH         = config['validation_LABEL_PATH']
fold_weights                  = config['fold_weights']
out_file                      = config['out_file']
input_shape                   = config['input_shape']
num_classes                   = config['num_classes']
activation_function           = config['activation_function']


############################GPU ###########################################
gpus = tf.config.experimental.list_physical_devices('GPU')  ## If you have gpu
#tf.config.experimental.set_memory_growth(gpus[0], True)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#print(os.getenv('TF_GPU_ALLOCATOR'))



########################## Train and Validation data ######################
max_score = 48
min_score  = 2 


low = (10-(min_score))/(max_score-min_score)
mid = (25-(min_score))/(max_score-min_score)

############################################################################
# Training data (total_score)

val_accuracy_dict  = {}
train_accuracy_dict = {}
val_loss_dict = {}
train_loss_dict = {}

for val_s in validation_slide:
    tf.keras.backend.clear_session()
    
    val_dir  = val_s

    train_DATA_PATH = val_dir+train_DATA_PATH
    validation_DATA_PATH = val_dir+validation_DATA_PATH

    input_shape = (224,224,3)

    train_images      = glob.glob(train_DATA_PATH+'*.tif')
    validation_images = glob.glob(validation_DATA_PATH+'*.tif')

    n_t= len(train_images)
    print('Train data size is {}'.format(n_t))
    with tf.device('/GPU:0'):
        train_samples = np.zeros((n_t, input_shape[0], input_shape[1], 3), dtype=np.float32)
        train_label   = np.zeros((n_t,), int)


        with open(val_dir+train_LABEL_PATH,'r') as f:
            for ctr,line in zip(range(n_t),f):
                if io.imread(train_DATA_PATH+(line.split()[0])) is not None:
                    train_samples[ctr,:,:,:] = io.imread(train_DATA_PATH+(line.split()[0]))
                    la = float(line.split(' ')[1])
                    if la > mid:
                        train_label[ctr] = 1
                    elif la > low:
                        train_label[ctr] = 2
                    else:
                        train_label[ctr] = 0
                else:
                    print(line.split(' ')[0],'is None')
            
        #Extreme labeling based on threshold
        new_train_label=train_label[(train_label!=2)]
        print('new_labels are:',new_train_label)
        new_train_samples=train_samples[(train_label!=2),:,:,:]
        print('new_samples are:',new_train_samples.shape)        
     
        
        for i in range(len(new_train_label)):
            mod_img = new_train_samples[i,:, :, :].astype(np.float32)
            new_train_samples[i, :, :, :] = mod_img

        #print(samples.dtype)
        new_train_samples = np.clip(new_train_samples, 0, 255)
        #train_samples /= 255
        #train_samples -= 0.5
        print(val_dir,'trainnnnn  is done!!')  
        
        ################################################################################
        # Validation dataset
        n_v= len(validation_images)
        print('validation data size is {}'.format(n_v))
        validation_samples = np.zeros((n_v, input_shape[0], input_shape[1], 3), dtype=np.float32)
        validation_label   = np.zeros((n_v,), int)

        with open(val_dir + validation_LABEL_PATH,'r') as f:
            for ctr,line in zip(range(n_v),f):
                if cv2.imread(validation_DATA_PATH+(line.split()[0])) is not None:
                    validation_samples[ctr,:,:,:]=cv2.imread(validation_DATA_PATH+(line.split()[0]))
                    la = float(line.split()[1])
                    if la > mid:
                        validation_label[ctr] = 1
                    elif la > low:
                        validation_label[ctr] = 2
                    else:
                        validation_label[ctr] = 0
                else:
                    print(line.split(' ')[0],'is Noneeee')
        new_validation_label  = validation_label[(validation_label!=2)]
        new_validation_samples= validation_samples[(validation_label!=2),:,:,:]
        print('new_validation_label:',new_validation_label)        
        #validation_label = tf.one_hot(validation_label,depth = 3)  

        print(val_dir,'is done!!')  
        for i in range(len(new_validation_label)):
            mod_img = new_validation_samples[i,:, :,:].astype(np.float32)
            new_validation_samples[i, :, :, :] = mod_img

        new_validation_samples = np.clip(new_validation_samples, 0, 255)
    #validation_samples /= 255
    #validation_samples -= 0.5
    print(new_validation_samples.shape)
    print(new_validation_label.shape)
    ################################################################################
    # Augmentation
    augmentor = ImageAugmentor()  # Instantiate the ImageAugmentor class 

    # Use augmentor.flow to create an augmented data generator
    train_data_generator = augmentor.flow(new_train_samples, new_train_label, batch_size=16)

    
    #############################################################
	## Training the model
    model = pretrained_model(new_train_samples.shape[1:], num_classes,activation_function ) ## REGRESSION: number of classes 1 instead of 3
    model.summary()


    for layer in model.layers:
        print(layer, layer.trainable)
   
    mcp_save = ModelCheckpoint(val_dir+'_best_weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5', monitor='val_sparse_categorical_accuracy', mode='max')

    performance_viz_cbk = PerformanceVisualizationCallback(
                                       model=model,
                                       validation_data=(validation_samples, validation_label),
                                       image_dir=val_dir+'_perorfmance_charts')

    my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath=val_dir+'_best_weights.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.hdf5')]#,
    #tf.keras.callbacks.TensorBoard(log_dir=val_dir+'_logs'),
    #performance_viz_cbk]
      
    with tf.device('/GPU:0'):
        # Train the model
        hist = model.fit(
        train_data_generator,  # Use the custom data generator
        epochs=25, 
        callbacks=my_callbacks,
        validation_data=(new_validation_samples, new_validation_label),
        validation_batch_size=16
    )   
        
    print("Training done for validation set of <<{}>>!".format(val_dir))
    model.save_weights('final_weights_'+val_dir+'.h5')
    df = pd.DataFrame.from_dict(hist.history, orient="index")
    df.to_csv(val_dir+"_raw.csv")
    ###########PLOT training curve##################################################################
   
    from plot import multi_plot
   
    multi_plot(None, [hist.history['loss'], hist.history['val_loss']],val_dir+'Loss.png', 'Epochs', 'Loss', legend=['Training', 'Validation'])
    multi_plot(None, [hist.history['multiclass_true_positives'], hist.history['val_multiclass_true_positives']],val_dir+'TP.png', 'Epochs', 'True positive', legend=['Training', 'Validation'])
    multi_plot(None, [hist.history['sparse_categorical_accuracy'], hist.history['val_sparse_categorical_accuracy']],val_dir+'_ACC.png', 'Epochs', 'Accuracy', legend=['Training', 'Validation'])


###################################################################################################
    
####################################################################################################
        
    
        
   




