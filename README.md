# brain-tumor-classification


In this project, I have trained several network-based machine learning models to detect three types of tumors in MRI images. The models are built using the keras python library. Two of the models are simple neural networks, and three are more complex convolutional neural networks. The networks are contained in the notebooks, and the data can be found on Kaggle [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

To use the models, first download the data and unzip it into the folder containing the notebooks. Then, rename the "Training" and "Testing" subfolders in the data to "train" and "test". Then, run `preprocessing.py`. You will get a folder named "cleaned". Delete the original data folder and rename the "cleaned" folder to "data". Now you can use the notebooks. The smaller models train in around 5-10 minutes, and the larger ones may take several hours. For this reason, I recommend using a cloud service like AWS or Azure to train the models.
