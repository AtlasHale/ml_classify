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
import numpy as np
import csv


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


def sliced_data(subset_percentage, project_home):
    classes = [folder for folder in os.listdir(os.path.join(project_home, 'data', 'train'))]
    counter = 0
    for folder in classes:
        image_number = (subset_percentage / 100) * len(os.listdir(os.path.join(project_home, 'data', 'train', folder)))
        if not image_number.is_integer():
            image_number = int(image_number) + 1
        if image_number == 0:
            continue
        images = [image for image in os.listdir(os.path.join(project_home, 'data', 'train', folder))]
        os.mkdir(os.path.join(project_home, 'data', 'temp', folder))
        uniques = set()
        for i in range(int(image_number)):
            index = np.random.randint(0, len(images))
            while index in uniques:
                index = np.random.randint(0, len(images))
            dst = os.path.join(project_home, 'data', 'temp', folder)
            src = os.path.join(project_home, 'data', 'train', folder, images[index])
            shutil.copy(src, dst)
            end = str(np.random.randint(0,100000))
            shutil.move((os.path.join(dst, images[index])), os.path.join(dst, end+images[index]))
            counter += 1
    with tarfile.open(os.path.join(project_home, 'data', str(subset_percentage)+'_train.tar.gz'), 'w:gz') as tar:
        tar.add(os.path.join(project_home, 'data', 'temp'), arcname=os.path.basename(os.path.join(project_home, 'data', 'temp')))
    tar.close()


if __name__ == '__main__':

    # train the algorithm on incrementally increasing amounts of training data
    percent = [i for i in range(2, 102, 2)]
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
    project_home = os.environ.get('PROJECT_HOME')
    if os.path.exists(os.path.join(project_home, 'data')):
        shutil.rmtree(os.path.join(project_home, 'data'))
    if not os.path.exists(os.path.join(project_home, 'data')):
        os.mkdir(os.path.join(project_home, 'data'))

    utils.unpack(project_home, args.train_tar, learning_curve=True)
    utils.unpack(project_home, args.val_tar, learning_curve=True)

    train_dir = os.path.join(project_home, 'data', 'train')
    val_dir = os.path.join(project_home, 'data', 'val')

    csv_data = [['percent', 'acc']]
    if os.path.exists(os.path.join(project_home, 'train.csv')):
        os.remove(os.path.join(project_home, 'train.csv'))
    with open(os.path.join(project_home, 'train.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
    csv_file.close()

    csv_data = [['percent', 'val']]
    if os.path.exists(os.path.join(project_home, 'val.csv')):
        os.remove(os.path.join(project_home, 'val.csv'))
    with open(os.path.join(project_home, 'val.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
    csv_file.close()

    folders = sorted(os.listdir(os.path.join(project_home, 'data', 'train')))
    print(folders)
    for label in folders:
        filename = label + '.csv'
        csv_data = [['precision', 'recall', 'F1']]
        if os.path.exists(os.path.join(project_home, filename)):
            os.remove(os.path.join(project_home, filename))
    headers = ['percent']
    for f in folders:
        headers.append(f)
    csv_data = [headers]
    with open(os.path.join(project_home, 'image_numbers.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
    csv_file.close()

    for p in percent:
        # subsample
        if os.path.exists(os.path.join(project_home, 'best.weights.hdf5')):
            shutil.rmtree(os.path.join(project_home, 'best.weights.hdf5'))
        temp_dir = os.path.join(project_home, 'data', 'temp')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        sliced_data(p, project_home)
        # subset_train_tar = temp_dir
        # size = p

        # # replace training with subset
        args.train_tar = os.path.join(project_home, 'data', 'temp.tar.gz')
        # train and store history of results
        model = Train().train_model(args)
        # hist_dict[p] = model
        # training_size[p] = size
        idx = 0
        for i in range(len(model.history['val_categorical_accuracy'])):
            if i == 0:
                maxValAcc = model.history['val_categorical_accuracy'][0]
            else:
                if model.history['val_categorical_accuracy'][i] > maxValAcc:
                    maxValAcc = model.history['val_categorical_accuracy'][i]
                    idx = i
        csv_data = []
        row = [p]
        for f in folders:
            row.append(len(os.listdir(os.path.join(project_home, 'data', 'temp', f))))
        csv_data.append(row)
        with open(os.path.join(project_home, 'image_numbers.csv'), 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_data)
        csv_file.close()

        wandb.log({"train_error": 1 - model.history['categorical_accuracy'][idx], "Percent Train Data": p})
        wandb.log({"val_error": 1 - model.history['val_categorical_accuracy'][idx], "Percent Train Data": p})

        csv_data = [[p, model.history['categorical_accuracy'][idx]]]
        with open(os.path.join(project_home, 'train.csv'), 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_data)
        csv_file.close()

        csv_data = [[p, model.history['val_categorical_accuracy'][idx]]]
        with open(os.path.join(project_home, 'val.csv'), 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_data)
        csv_file.close()
    # plot the last error of each training cycle and log as object in wandb
    # this will convert to plotly by default in wandb

