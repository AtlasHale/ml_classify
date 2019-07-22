Steps to run should be:

cd ~
git pull https://github.com/AtlasHale/ml_classify   
cd ml_classify  
Upload google service credentials as credentials.json   
https://cloud.google.com/storage/docs/reference/libraries to find how to generate API key   



Export project home as $PWD     
Export tar bucket as name of bucket with tar files      
Export wandb run group, user, api key   
Export google service credentials location ($PWD/credentials.json)  

Run training    
python3 src/train.py --horizontal_flip True --augment_range 0.2 --train_tar train.tar.gz --val_tar val.tar.gz --lr 0.001 --base_model inceptionv3 --project inception_training --batch_size 4 --epoch 50

Run learning rate   
python3 src/learning_curve.py --horizontal_flip True --augment_range 0.2 --train_tar 100_train.tar.gz --val_tar val.tar.gz --lr 0.001 --base_model inceptionv3 --project inception_learning_curve --batch_size 4 --epoch 10

