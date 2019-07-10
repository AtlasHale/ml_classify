To run:

python3 src/train.py --horizontal_flip True --augment_range 0.2 --train_tar 
data/train --val_tar data/val --lr 0.001  --base_model vgg16 --project fathomnet --batch_size 4 --epoch 5

you should have non tar.gz folders named train and val in your data folder.    

export WANDB credentials before running, or edit ~/.bashrc to permanently set.    
