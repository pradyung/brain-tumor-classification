import os
import cv2
import numpy as np
from keras.utils import to_categorical

def load_data():
    tumors = ['glioma', 'meningioma', 'pituitary', 'notumor']
    tumor_dict = {'glioma': 0, 'meningioma': 1, 'pituitary': 2, 'notumor': 3}

    final_data = {
        'train': {'x': [], 'y': []},
        'test': {'x': [], 'y': []}
    }

    for dataset in ['train', 'test']:
        for tumor in tumors:
            for i in os.listdir(f'data/{dataset}/{tumor}'):
                image = cv2.imread(f'data/{dataset}/{tumor}/{i}')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (256, 256))
                final_data[dataset]['x'].append(image)
                final_data[dataset]['y'].append(tumor_dict[tumor])

    x_train = np.array(final_data['train']['x']).astype('float32') / 255
    x_test = np.array(final_data['test']['x']).astype('float32') / 255
    
    y_train = to_categorical(np.array(final_data['train']['y']))
    y_test = to_categorical(np.array(final_data['test']['y']))

    return x_train, y_train, x_test, y_test