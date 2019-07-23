Steps to run should be:

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

Run training    
python3 src/train.py --horizontal_flip True --augment_range 0.2 --train_tar train.tar.gz --val_tar val.tar.gz --lr 0.001 --base_model inceptionv3 --project inception_training --batch_size 4 --epoch 50

Run learning rate   
python3 src/learning_curve.py --horizontal_flip True --augment_range 0.2 --train_tar 100_train.tar.gz --val_tar val.tar.gz --lr 0.001 --base_model inceptionv3 --project inception_learning_curve --batch_size 4 --epoch 10


Google Cloud Environment Set-Up     
Go to https://cloud.google.com/     
In the upper right, sign in to your google account. 
In the upper right, click console.  
Create a new project by selecting the drop down menu in the upper left. 
A pop-up window will have a new projects button in the upper right.     
Select your new project from the dropdown menu to make it your active project. 
