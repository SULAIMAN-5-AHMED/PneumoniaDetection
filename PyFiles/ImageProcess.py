"""
{0:Normal
1: Pneumonia
}
"""


import os
import pandas as pd
import numpy as np
import cv2 as cv

def folder_to_data(file_path):
    data = []
    for folder in os.listdir(file_path):
        for img in os.listdir(os.path.join(file_path,folder)):
            data.append({'Type' : folder,
            'item' : img
            })
    data = pd.DataFrame(data)
    data = data.sample(frac=1)
    return data

data = folder_to_data("/chest_xray/test")
print(len(data))

images = []
labels = []
for index,row in data.iterrows():
    label = row['Type']
    image = row['item']
    img = cv.imread(r"C:\\Users\\sulai\\Desktop\\PYTHON\\MedicalScan\\chest_xray\\test\\{}\\{}".format(label,image))
    """
    The images will now open sequentially and some changes would be made to them
    -> First we will convert them from BGR(default color channel) to RGB as most libraries supports the RGB
    -> Then we will resize it to 200*200 pixels 
    -> Convert them into an array and then append them to the list
    -> As for labels we will used the label_encoder.fit_transform(labels) before appending
    to the labels list
    -> We wills ave them in the form of .npy format to be later passed to Google Colab
    """
    img_formated = cv.resize(cv.cvtColor(img,cv.COLOR_BGR2GRAY),(200,200))
    max_pixl = 255.0 # since we have 255 color range of 0-255
    img_normalized = img_formated.astype('float32')/max_pixl
    img_arr = np.array(img_normalized)
    """
    In the above we added converted the image to an array and appended
    it to the images list
    """

    images.append(np.expand_dims(img_arr,axis=-1))
    labels.append(int(label))

"""Applied the encoder the whole list to maintain the consistency of the labels """
np.save("Test(x).npy",np.array(images))
np.save("Test(y).npy",np.array(labels))