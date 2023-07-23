import os
from PIL import Image
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
                image = Image.open(f'data/{dataset}/{tumor}/{i}')
                image = image.convert('L')
                image = image.resize((256, 256))

                final_data[dataset]['x'].append(np.array(image).reshape((256,256,1)))
                final_data[dataset]['y'].append(tumor_dict[tumor])

    x_train = np.array(final_data['train']['x']).astype('float32') / 255
    x_test = np.array(final_data['test']['x']).astype('float32') / 255
    
    y_train = to_categorical(np.array(final_data['train']['y']))
    y_test = to_categorical(np.array(final_data['test']['y']))

    np.random.seed(42)
    np.random.shuffle(x_train)
    np.random.seed(42)
    np.random.shuffle(y_train)

    print(y_train)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(128, (5, 5), activation='relu', input_shape=(256, 256, 1)))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, batch_size=32, epochs=15, verbose=1, validation_split=0.2)

model.save('result/model.h5')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('result/accuracy.png')

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('result/loss.png')

score = model.evaluate(x_test, y_test, verbose=1)

with open('result/score.txt', 'w') as f:
    f.write(str(dict(zip(model.metrics_names, score))))
