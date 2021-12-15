import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
import os
from sklearn.metrics import plot_confusion_matrix


# data loading
images = []
labels = []
files = os.listdir(r'.\processed_data2')
for  proimg in files:
    image = imread(f'.\\processed_data2\\{proimg}').ravel()/255
    label = int(proimg.split('.')[0][-1])
    images.append(image)
    labels.append(label)

Images = np.array([image for image in images])
Labels = np.array(labels)

from joblib import load
model = load('svm_classifier2')
ypred = model.predict(Images)

plot_confusion_matrix(model, Images, Labels)
plt.show()