import os
import utils
import argparser
import train
from train import Train
from argparser import ArgParser
import tempfile
import glob
import random
import shutil
import tarfile
import wandb

def subsample(subset_percentage, train_dir):
    file_list = []
    print(f'Looking for prediction images in {train_dir}')
    file_glob = os.path.join(train_dir, '**/*.jpg')
    file_list.extend(glob.iglob(file_glob, recursive=True))

    if subset_percentage < 100:
        print(f'Randomly selecting {subset_percentage} % of images')
        k = len(file_list) * subset_percentage // 100
        indicies = random.sample(range(len(file_list)), k)
        file_list = [file_list[i] for i in indicies]

    # get labels names inferred from directory names and create subdirectory for each
    # TODO: verify whether this needs to be sorted or not
    all_dirs = [x[0] for x in os.walk(train_dir)]
    labels = [os.path.basename(x) for x in all_dirs[1:len(all_dirs)]]
    temp_dir = tempfile.mkdtemp()
    for l in labels:
        os.makedirs(os.path.join(temp_dir, l))
    print(f'Copying results to {temp_dir}')

    for src in file_list:
        fname = os.path.basename(src)
        class_name = os.path.basename(os.path.normpath(os.path.dirname(src)))
        dst = f'{temp_dir}/{class_name}/{fname}'
        shutil.copy(src, dst)

    import uuid
    out_tar = str(uuid.uuid4()) + '.tar.gz'
    print(f'Compressing results to {out_tar}')
    with tarfile.open(out_tar, "w:gz") as tar:
        tar.add(temp_dir, arcname='.')

    # clean-up
    shutil.rmtree(temp_dir)

    return out_tar, len(file_list)

if __name__ == '__main__':

    # train the algorithm on incrementally increasing amounts of training data
    percent = [5, 20, 50, 75, 100]
    training_size = {}
    hist_dict = {}

    parser = ArgParser()
    args = parser.parse_args()

    # check connection to wandb and log basic parameters before starting
    env = os.environ.copy()
    if 'WANDB_RUN_GROUP' not in env:
        print('Need to set WANDB_RUN_GROUP environment variable for this run')
        exit(-1)
    print('Connecting to wandb with group {}'.format(env['WANDB_RUN_GROUP']))
    wandb.init(project=args.project, job_type='training', name='kerasclassification-' + args.project,
               dir=os.getcwd())

    parser.log_params(wandb)

    # unpack original images
    output_dir = tempfile.mkdtemp()
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    utils.unpack(train_dir, args.train_tar)
    utils.unpack(val_dir, args.val_tar)

    for p in percent:
        # subsample
        subset_train_tar, size = subsample(p, train_dir)
        print(f'Total images {size} in training data for subsample {p} %%')

        # replace training with subset
        args.train_tar = subset_train_tar

        # train and store history of results
        hist_dict[p] =  Train().train_model(args)
        training_size[p] = size

    # plot the last error of each training cycle and log as object in wandb
    # this will convert to plotly by default in wandb
    for percent, history in hist_dict.items():
        train_error = 1 - history.history['acc'][-1]
        val_error = 1 - history.history['val_acc'][-1]
        plt.plot(training_size[percent], train_error, 'ro')
        plt.plot(training_size[percent], val_error,'bo')
    plt.title('Learning curve')
    plt.xlabel('Training set size')
    plt.ylabel('Error')
    plt.legend()
    wandb.log({"learning curve": plt})