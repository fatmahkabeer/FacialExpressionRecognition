#%%
import pandas as pd
import numpy as np

import tensorflow as tf

from IPython.display import clear_output

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import roc_auc_score

import matplotlib as plt


# %%
train_dir = "archive/train"
test_dir = "archive/test"

#%%
import os
#number of categories. 
len(os.listdir(train_dir))
#%%
# target distrubuation. 
print('number of angry images: ' + str(len(os.listdir('archive/test/angry'))))
print('number of disgusted images: ' + str(len(os.listdir('archive/test/disgusted'))))
print('number of fearful images: ' + str(len(os.listdir('archive/test/fearful'))))
print('number of happy images: ' + str(len(os.listdir('archive/test/happy'))))
print('number of neutral images: ' + str(len(os.listdir('archive/test/neutral'))))
print('number of sad images: ' + str(len(os.listdir('archive/test/sad'))))
print('number of surprised images: ' + str(len(os.listdir('archive/test/surprised'))))

#%%
train_d= ImageDataGenerator(horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.05,
                                   rescale = 1./255,
                                   validation_split = 0.2,
                                   preprocessing_function=tf.keras.applications.densenet.preprocess_input
                                  )
test_d = ImageDataGenerator(rescale = 1./255,
                                  validation_split = 0.2,
                                  preprocessing_function=tf.keras.applications.densenet.preprocess_input)

train_g= train_d.flow_from_directory(directory = train_dir,
                                                    target_size = (48 ,48),
                                                    batch_size = 64,
                                                    shuffle  = True , 
                                                    color_mode = "rgb",
                                                    class_mode = "categorical",
                                                    subset = "training",
                                                    seed = 9)

validation_g= test_d.flow_from_directory(directory = train_dir,
                                                         target_size = (48 ,48),
                                                         batch_size = 64,
                                                         shuffle  = True , 
                                                         color_mode = "rgb",
                                                         class_mode = "categorical",
                                                         subset = "validation",
                                                         seed = 9
                                                        )

test_g= test_d.flow_from_directory(directory = test_dir,
                                                   target_size = (48, 48),
                                                    batch_size = 64,
                                                    shuffle  = False , 
                                                    color_mode = "rgb",
                                                    class_mode = "categorical",
                                                    seed = 9)


#%%    
inputs = tf.keras.layers.Input(shape=(48 ,48,3))
feature = tf.keras.applications.DenseNet169(input_shape=(48,48, 3),
                                               include_top=False,
                                               weights="imagenet")(inputs)

classification_output = tf.keras.layers.GlobalAveragePooling2D()(feature)
classification_output = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(classification_output)
classification_output = tf.keras.layers.Dropout(0.3)(classification_output)
classification_output = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(classification_output)
classification_output = tf.keras.layers.Dropout(0.5)(classification_output)
classification_output = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(classification_output)
classification_output = tf.keras.layers.Dropout(0.5) (classification_output)
classification_output = tf.keras.layers.Dense(7, activation="softmax", name="classification")(classification_output)

model = tf.keras.Model(inputs=inputs, outputs = classification_output)
     
model.compile(optimizer=tf.keras.optimizers.SGD(0.1), 
                loss='categorical_crossentropy',
                metrics = ['accuracy'])
  


clear_output()

# Feezing the feature extraction layers
model.layers[1].trainable = False

model.summary()
# %%
earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                         patience=3,
                                                         verbose= 1 ,
                                                         restore_best_weights=True
                                                        )

history = model.fit(x = train_g,
                    epochs = 20,
                    validation_data = validation_g, 
                    callbacks= [earlyStoppingCallback])

history = pd.DataFrame(history.history)
# %%
# Un-Freezing the feature extraction layers for fine tuning 
model.layers[1].trainable = True

model.compile(optimizer=tf.keras.optimizers.SGD(0.001), #lower learning rate
                loss='categorical_crossentropy',
                metrics = ['accuracy'])

history_ = model.fit(x = train_g,epochs = 15 ,validation_data = validation_g)
history = history.append(pd.DataFrame(history_.history) , ignore_index=True)


#%%
model.evaluate(test_g)
preds = model.predict(test_g)
y_test = np.array(test_g.labels)
#%%
print("ROC-AUC Score  = " ,roc_auc_score(to_categorical(y_test) , preds))
