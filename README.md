# MA553-MagicTimes

## Basic Layout
The parts of the project in the local folder are to be run locally. The jupyter folder must be uploaded to an AWS Sagemaker notebook instance.

## Local Data Preparation
The project is started through running the script dataCleaning.py to serialize the data from mysql into a more usable format. Then, the files names universalData.py, disneyData.py, and seaworldData.py should be run to generate csv files with the data for each collection of parks.

## AWS Model Training
Once the data has been generated, the set of data for each grouping of parks must be synthesized. This is done in AWS Sagemaker Notebooks. For each collection of parks, run the data cleaning script to generate the pickled data, and then run the analyzer to generate the models.

## Local Model Analysis
Following model training, the test pickled data and completed models need to be downloaded to you local device. From there, the supplied analyzer scipt for the collection of parks can be run and accuracy metrics will be reported.
