# Keras Classification of Deep Sea Imagery
This repository contains code which can be used to generate a keras model to predict classes when given training data. The code utilizes transfer learning, working by using pre-trained deep neural net models such as inception and resnet, and aplpying them to a specific data set. This repository is primarily set up to be run on a Google Compute VM, however simple adjustments can be made to run on AWS, other cloud services, or locally.
##
### Building Google Cloud VM Environment
The steps to setting up a google cloud environment are the following:
* Create a new Google Cloud Project
* Create a cloud storage bucket
* Split data appropriately
* Upload data to cloud storage
* Create a Compute VM 
* Update drivers and Python version
* Clone Repository
* Update Environment Variables
* Install Required modules
##
### Create a Google Cloud Project

Google Cloud Environment Set-Up     
Go to https://cloud.google.com/     
In the upper right, sign in to your google account. 
In the upper right, click console.  
Create a new project by selecting the drop down menu in the upper left. 
A pop-up window will have a new projects button in the upper right.     
Select your new project from the dropdown menu to make it your active project. 
##
### Create Cloud Storage Bucket

In the search bar, search bucket and select "create bucket".
Assign your bucket a unique name.    
Select regional storage class.  
Set object-level and bucket-level permissions.  
Create bucket.  
##
###  Split Data

Assuming your data is seperated by class into folders, but not into a train/test/val set    
Run `pip install split-folders` 
Open a terminal and run python interactively using `$ python3`  
`import split_folders`  
`split_folders.ratio('input_folder', output='output_folder', seed=1337, ratio=(.8, .1, .1))`    
Read about [split-folders](https://pypi.org/project/split-folders/) 
##
### Running Inference 
To run general Inference with basic metrics:
```
python3 src/train.py --horizontal_flip True --augment_range 0.2 \
--train_tar train.tar.gz --val_tar val.tar.gz --lr 0.001 --base_model inceptionv3 \
--project inception_training --batch_size 4 --epoch 50
```
To run incrementally increasing the size of the training data set per class:
```
python3 src/learning_curve.py --horizontal_flip True --augment_range 0.2 \
--train_tar 100_train.tar.gz --val_tar val.tar.gz --lr 0.001 --base_model inceptionv3 \
--project inception_learning_curve --batch_size 4 --epoch 10
```
##
cd ~  
git clone https://github.com/AtlasHale/ml_classify     
cd ml_classify    
Upload google service credentials as credentials.json   
mv ~/credentials.json ~/ml_classify/  
https://cloud.google.com/storage/docs/reference/libraries to find how to generate API key   



Export project home as $PWD     
Export tar bucket as name of bucket with tar files      
Export wandb run group, user, api key   
Export google service credentials location ($PWD/credentials.json)  

