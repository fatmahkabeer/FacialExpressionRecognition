# FacialExpressionRecognition
# This is done with Kaggle's notbook assistance. 

1. Get Data
I got the the data from Kaggle, I coudn't uplaod it though because of the file size. 
https://www.kaggle.com/ananthu017/emotion-detection-fer
The dataset has 7 folders for training (happiness, neutral, sadness, anger, surprise, disgust, fear) and 7 folders for testing (happiness, neutral, sadness, anger, surprise, disgust, fear).
It has 5,685 examples of 48x48 pixel gray scale images.

2. Clean, Prepare & Manipulate Data
• I used the keras perprocess function for preparing images. tf.keras.applications.densenet.preprocess_input
• The data already split into testing and training folders.

3. Train Model
• Extracting features using DenseNet
• Then train the model using 4 denis layers.
• SGD was used for optimizers categorical_crossentropy as a loss function

4. Test Model
The model gives an accuracy of 0.8910888723078322
