import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#data pre-pocceing
data = pd.read_csv("data\clothing-dataset\images.csv")
print(data.head())
#1.load csv into a dataframe
#2.remove unrellevt coulums
#3.remove rows with undefind lable
#4.create dataset of images and labels
#4.1.create tensor of the data
#4.2.replace serial numbers with images

data = data.drop('sender_id', 1)
data = data.drop('kids', 1)
print(data.head())