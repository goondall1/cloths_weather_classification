import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split

#data pre - processing
#1.load csv into a dataframe
data = pd.read_csv("./data/clothing-dataset/images.csv")

#2.remove unrellevt coulums
print(data.head())
data = data.drop('sender_id', 1)
data = data.drop('kids', 1)
print(data.head())

#3.remove rows with undefind lable
print(data[data['label']=='Not sure'].count())
data = data[data.label != 'Not sure']
print(data.head())
print(data[data['label']=='Not sure'].count())  #FIXME: remove also 'Other' label.

#4.create dataset of images and labels
#4.1.create tensor of the data
#4.2.replace serial numbers with images


#loading an image to tensorflow
new_path = "./data/clothing-dataset/images/0a3e62e3-fac5-4648-9da2-f6bc4074ee31.jpg"
load_image = tf.keras.preprocessing.image.load_img(os.path.join(new_path))
print(load_image)

max_size_l=[]
for image_str in data["image"]:
    image_str_with_sofix = image_str + '.jpg'
    new_path = "./data/clothing-dataset/images/" + image_str_with_sofix
    try:
        #load_image = tf.keras.preprocessing.image.load_img(os.path.join(new_path))
        load_image = tf.io.read_file(filename=new_path) # loading image to a variable
        load_image = tf.image.decode_jpeg(load_image, channels=3)  # transforming the image to a tensor
        max_size_l.append(load_image.shape)
        #print(load_image)
        #FIXME: add all tensors to df.
    except :
        data = data[data.label != image_str]
        continue

# print(max_size_l)
# print(max(max_size_l)) #TODO: add me later

labels = list(data['label'].unique())
print(labels)
repl={}
for i in range(len(labels)):
    repl[labels[i]]=i
print(repl)
data.replace(repl,inplace=True)
print(data.head())

train_df,test_df=train_test_split(data,test_size=0.2)
print(len(train_df))
print(len(test_df))
train_df.to_csv('./data/train.csv',index=False,header=True)
test_df.to_csv('./data/test.csv',index=False,header=True)
print(train_df.head())
#creating train dataset
#transforming to tesor





# model=tf.keras.Sequential([
#                            tf.keras.layers.Flatten(input_shape=(28,28)),
#                            tf.keras.layers.Dense(250,activation='relu'),
#                            tf.keras.layers.Dense(18)
# ])
# model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=20)
# probability_model=tf.keras.Sequential([model,tf.keras.layers.Softmax()])
# predictions=probability_model.predict(x_test)
# predictions[0]
# np.argmax(predictions[0])
# plt.imshow(x_test[0])
# plt.show()


