__author__ = 'Connor Hale, Danelle Cline, Nathanial Yee'

"""
Taken from code provided by Danelle Cline at MBARI, code updated from cats and dogs to match a deep sea data set.
"""


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print('Adding {} to path'.format(parentdir))
import tensorflow as tf
import tempfile
import keras
from keras import optimizers
from keras import metrics
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from transfer_model import TransferModel
import wandb
from wandb.keras import WandbCallback
from argparser import ArgParser
import utils
from plot import Plot
import glob
import conf

class Train():

    def __init__(self):
        return

    def compile_and_fit_model(self, model, fine_tune_at, train_generator, validation_generator, epochs,
                              batch_size, loss, optimizer, lr, labels,
                              metrics=metrics.categorical_accuracy,
                              save_model=False, output_dir='/tmp'):

        print('Writing TensorFlow events locally to {}'.format(output_dir))
        tensorboard = TensorBoard(log_dir=output_dir)

        steps_per_epoch = train_generator.n // batch_size
        validation_steps = validation_generator.n // batch_size

        # Un-freeze the top layers of the model
        model.trainable = True

        # if fine tune at defined, freeze all the layers before the `fine_tune_at` layer
        if fine_tune_at > 0:
            for layer in model.layers[:fine_tune_at]:
                layer.trainable = False

        if optimizer == 'rmsprop':
                opt = optimizers.RMSprop(lr=lr)
                model.compile(optimizer=opt,
                              loss=loss,
                              metrics=[metrics])
        elif optimizer == 'adam':
            model.compile(loss=loss,
                          optimizer=tf.keras.optimizers.Adam(lr=lr),
                          metrics=[metrics])
        else:
            model.compile(loss=loss,
                          optimizer=tf.keras.optimizers.SGD(lr=lr),
                          metrics=[metrics])

        if loss == 'categorical_crossentropy':
            monitor = 'val_categorical_accuracy'
        else:
            monitor = 'val_binary_accuracy'

        early = EarlyStopping(monitor=monitor, min_delta=0, patience=2, verbose=1, mode='auto')
        checkpoint_path = '{}/checkpoints.best.h5'.format(output_dir)
        checkpoint = ModelCheckpoint(checkpoint_path, monitor=monitor, verbose=1, save_best_only=True, mode='max')
        if os.path.exists(checkpoint_path):
            print('Loading model weights from {}'.format(checkpoint_path))
            model.load_weights(checkpoint_path)

        '''schedule = SGDRScheduler(min_lr=conf.MIN_LR,
                                 max_lr=conf.MAX_LR,
                                 steps_per_epoch=np.ceil(epochs / batch_size),
                                 lr_decay=0.9,
                                 cycle_length=5,
                                 mult_factor=1.5)'''

        history = model.fit_generator(train_generator,
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           use_multiprocessing=True,
                                           validation_data=validation_generator,
                                           validation_steps=validation_steps,
                                           callbacks=[tensorboard, checkpoint, early,
                                                      WandbCallback(data_type="image", labels=labels)])# , schedule])

        if save_model:
            model_dir = self.get_directory_path("keras_models")
            self.keras_save_model(model, model_dir)

        return history

    def keras_save_model(self, model, model_dir='/tmp'):
        """
        Convert Keras estimator to TensorFlow
        :type model_dir: object
        """
        print("Model is saved locally to %s" % model_dir)
        mlflow.keras.save_model(model, model_dir)


    def evaluate_model(self,model, x_test, y_test):
        """
        Evaluate the model with unseen and untrained data
        :param model:
        :return: results of probability
        """
        return model.evaluate(x_test, y_test)

    def get_validation_loss(self, hist):
        val_loss = hist.history['val_loss']
        val_loss_value = val_loss[len(val_loss) - 1]
        return val_loss_value

    def get_validation_acc(self, hist):
        print("keys {}".format(hist.history.keys()))
        if 'val_binary_accuracy' in hist.history.keys():
            val_acc = hist.history['val_binary_accuracy']
        else:
            val_acc = hist.history['val_categorical_accuracy']
        val_acc_value = val_acc[len(val_acc) - 1]
        return val_acc_value

    def print_metrics(self, hist):
        if 'val_binary_accuracy' in hist.history.keys():
            acc_value = self.get_binary_acc(hist)
            loss_value = self.get_binary_loss(hist)
            print("Final metrics: binary_loss:%6.4f".format(loss_value))
            print("Final metrics: binary_accuracy=%6.4f".format(acc_value))

        val_acc_value = self.get_validation_acc(hist)
        val_loss_value = self.get_validation_loss(hist)

        print("Final metrics: validation_loss:%6.4f".format(val_loss_value))
        print("Final metrics: validation_accuracy:%6.4f".format(val_acc_value))

    def train_model(self, args):
        """
        Train the model and log all the metrics
        :param parser: command line argument object
        """

        sess = tf.InteractiveSession()

        # Rescale all images by 1./255 and apply image augmentation if requested
        train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                     width_shift_range=args.augment_range,
                                                                     height_shift_range=args.augment_range,
                                                                     zoom_range=args.augment_range,
                                                                     horizontal_flip=args.horizontal_flip,
                                                                     vertical_flip=args.vertical_flip,
                                                                     shear_range=args.shear_range
                                                                     )

        val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                   width_shift_range=args.augment_range,
                                                                   height_shift_range=args.augment_range,
                                                                   zoom_range=args.augment_range,
                                                                   horizontal_flip=args.horizontal_flip,
                                                                   vertical_flip=args.vertical_flip,
                                                                   shear_range=args.shear_range
                                                                   )

        output_dir = tempfile.mkdtemp()
        train_dir = os.path.join(output_dir, 'train')
        val_dir = os.path.join(output_dir, 'val')
        image_dir = os.path.join(output_dir, 'images')
        os.makedirs(image_dir)
        utils.unpack(train_dir, args.train_tar)
        utils.unpack(val_dir, args.val_tar)
        labels = list(os.listdir('/home/cohale/keras-classify/data/train'))

        model, image_size, fine_tune_at  = TransferModel(args.base_model).build(args.l2_weight_decay_alpha)
        train = Train()

        # Flow training images in batches of <batch_size> using train_datagen generator
        training_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=args.batch_size,
            class_mode='categorical')

        # Flow validation images in batches of <batch_size> using test_datagen generator
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(image_size, image_size),
            batch_size=args.batch_size,
            class_mode='categorical')
        model.summary()
        history = train.compile_and_fit_model(model=model, fine_tune_at=fine_tune_at,
                                              train_generator=training_generator, lr=args.lr,
                                              validation_generator=validation_generator,
                                              epochs=args.epochs, batch_size=args.batch_size,
                                              loss=args.loss, output_dir=output_dir,
                                              optimizer=args.optimizer,
                                              metrics=metrics.categorical_accuracy,
                                              labels=labels)
        train.print_metrics(history)

        # terminate tensorboard sessions
        sess.close()

from time import time

if __name__ == '__main__':

    parser = ArgParser()
    args = parser.parse_args()

    # Check connection to wandb  before starting
    env = os.environ.copy()

    # Initialize wandb
    if 'WANDB_RUN_GROUP' not in env:
        print('Need to set WANDB_RUN_GROUP environment variable for this run')
        exit(-1)
    print('Connecting to wandb with group {}'.format(env['WANDB_RUN_GROUP']))
    wandb.init(project=args.project, job_type='training', name='kerasclassification-' + args.project,
               dir=os.getcwd())

    parser.log_params(wandb)

    start_time = time()

    print("Using parameters")
    parser.summary()

    Train().train_model(args)

    runtime = time() - start_time

    print('Model complete. Total runtime {}'.format(runtime))

