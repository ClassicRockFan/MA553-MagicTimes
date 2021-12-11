#Necessary Imports
import argparse, os
import boto3
import numpy as np
import pandas as pd
import sagemaker
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import pickle


if __name__ == '__main__':
    
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    
    #Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    #sm_model_dir: model artifacts stored here after training
    #training directory has the data for the model
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default="s3://sagemaker-park-data/universal-data/universal_train.pkl")
    parser.add_argument('--test', type=str, default="s3://sagemaker-park-data/universal-data/universal_test.pkl")
    
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    lr         = args.learning_rate
    model_dir  = args.model_dir
    sm_model_dir = args.sm_model_dir
    training_dir   = args.train
    testing_dir = args.test

    s3 = boto3.resource('s3')    
    try:
        s3.Bucket("sagemaker-park-data").download_file("universal-data/universal_train.pkl", "universal_train.pkl")
        s3.Bucket("sagemaker-park-data").download_file("universal-data/universal_test.pkl", "universal_test.pkl")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            print(e)
            raise
    ############
    #Reading in data
    ############
    with open("universal_train.pkl",'rb') as f:
        train_data = pickle.load(f)
    with open("universal_test.pkl",'rb') as f:
        test_data = pickle.load(f)  
    
    ############
    #Preprocessing data
    ############

    x_train = train_data["x"][0:12000]
    y_train = train_data["y"][0:12000]
    x_test = test_data["x"]
    y_test = test_data["y"]
    
    ###########
    #Model Building
    ###########


    ts_inputs = Input(shape=x_train[0].shape)
    x = BatchNormalization()(ts_inputs)
    x = LSTM(units=128, return_sequences=False, activation='linear')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='tanh')(x)
    x = Dense(128, activation='tanh')(x)
    x = Dense(64, activation='tanh')(x)
    x = Dropout(0.2)(x)
    x = Dense(units=y_train.shape[1] * y_train.shape[2], activation='linear')(x)
    outputs = Reshape(target_shape=y_train[0].shape)(x)
    model = Model(inputs=ts_inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    model.summary()
    early_stop = EarlyStopping(patience=10)
    model.fit(x=x_train, 
          y=y_train, 
          epochs=epochs,
          validation_data=(x_test, y_test), verbose=1 ,callbacks=[early_stop])

    #Storing model artifacts
    model.save(os.path.join(sm_model_dir, "universal"), 'universal_model.h5')