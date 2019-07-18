add GOOGLE_APPLICATION_CREDENTIALS to env      
https://cloud.google.com/storage/docs/reference/libraries to find how to generate API key

example environment variables:    
WANDB_RUN_GROUP=mbari    
PROJECT_HOME=/Users/chale/Desktop/ml_classify    
GOOGLE_APPLICATION_CREDENTIALS=/Users/chale/Desktop/ml_classify/credentials.json
BUCKET_NAME=data_bucket_mbari    
TAR_BUCKET=s2019_tar_mbari    
WANDB_API_KEY=<api_key>
    
To run:   
``   
python3 src/train.py --horizontal_flip True --augment_range 0.2 --train_tar 
train.tar.gz --val_tar val.tar.gz --lr 0.001  --base_model inceptionv3 --project fathomnet --batch_size 4 --epoch 5
``

