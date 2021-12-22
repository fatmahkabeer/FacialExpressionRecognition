#%%
import pandas as pd
import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import roc_auc_score
# %%
train_dir = "archive/train"
test_dir = "archive/test"


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
                                                    seed = 12)

validation_g= test_d.flow_from_directory(directory = train_dir,
                                                         target_size = (48 ,48),
                                                         batch_size = 64,
                                                         shuffle  = True , 
                                                         color_mode = "rgb",
                                                         class_mode = "categorical",
                                                         subset = "validation",
                                                         seed = 12
                                                        )

test_g= test_d.flow_from_directory(directory = test_dir,
                                                   target_size = (48, 48),
                                                    batch_size = 64,
                                                    shuffle  = False , 
                                                    color_mode = "rgb",
                                                    class_mode = "categorical",
                                                    seed = 12)

#%%
def feature_extractor(inputs):
    feature_extractor = tf.keras.applications.DenseNet169(input_shape=(48,48, 3),
                                               include_top=False,
                                               weights="imagenet")(inputs)
    
    return feature_extractor

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer = tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dropout(0.5) (x)
    x = tf.keras.layers.Dense(7, activation="softmax", name="classification")(x)
    
    return x

def final_model(inputs):
    densenet_feature_extractor = feature_extractor(inputs)
    classification_output = classifier(densenet_feature_extractor)
    
    return classification_output

def define_model():
    
    inputs = tf.keras.layers.Input(shape=(48 ,48,3))
    classification_output = final_model(inputs) 
    model = tf.keras.Model(inputs=inputs, outputs = classification_output)
     
    model.compile(optimizer=tf.keras.optimizers.SGD(0.1), 
                loss='categorical_crossentropy',
                metrics = ['accuracy'])
  
    return model
#%%
from IPython.display import clear_output
# %%
model = define_model()
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
                    epochs = 30,
                    validation_data = validation_g, 
                    callbacks= [earlyStoppingCallback])

history = pd.DataFrame(history.history)
# %%
# Un-Freezing the feature extraction layers for fine tuning 
model.layers[1].trainable = True

model.compile(optimizer=tf.keras.optimizers.SGD(0.001), #lower learning rate
                loss='categorical_crossentropy',
                metrics = ['accuracy'])

history_ = model.fit(x = train_g,epochs = 20 ,validation_data = validation_g)
history = history.append(pd.DataFrame(history_.history) , ignore_index=True)


#%%
model.evaluate(test_g)
preds = model.predict(test_g)
y_preds = np.argmax(preds , axis = 1 )
y_test = np.array(test_g.labels)
#%%
print("ROC-AUC Score  = " ,roc_auc_score(to_categorical(y_test) , preds))
