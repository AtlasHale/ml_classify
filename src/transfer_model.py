import tensorflow as tf
from tensorflow import keras
import conf as model_conf

class TransferModel():

    def __init__(self, base_model_name):
        self.base_model_name = base_model_name
        if base_model_name not in model_conf.MODEL_DICT.keys():
            raise('{} not in {}'.format(base_model_name,model_conf.MODEL_DICT.keys()))
        return

    def build(self, l2_weight_decay_alpha = 0.):
        '''
        Build base model from the pre-trained model
        :param l2_weight_decay_alpha:
        :return: a Keras network model
        '''
        cfg = eval("model_conf.MODEL_DICT['{}']".format(self.base_model_name))
        image_size = cfg['image_size']
        IMG_SHAPE = (image_size, image_size, 3)
        model_instance = cfg['model_instance']

        base_model = eval("{}(input_shape={},include_top=False,weights='imagenet')".format(model_instance, IMG_SHAPE))

        if l2_weight_decay_alpha > 0.:
            if cfg['has_depthwise_layers']:
                for layer in base_model.layers:
                    if isinstance(layer, keras.layers.DepthwiseConv2D):
                        layer.add_loss(keras.regularizers.l2(l2_weight_decay_alpha)(l.depthwise_kernel))
                    elif isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                        layer.add_loss(keras.regularizers.l2(l2_weight_decay_alpha)(layer.kernel))
                    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                        layer.add_loss(keras.regularizers.l2(l2_weight_decay_alpha)(layer.bias))
            else:
                for layer in base_model.layers:
                    if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
                        layer.add_loss(keras.regularizers.l2(l2_weight_decay_alpha)(layer.kernel))
                    if hasattr(layer, 'bias_regularizer') and layer.use_bias:
                        layer.add_loss(keras.regularizers.l2(l2_weight_decay_alpha)(layer.bias))


        # Freezing (or setting layer.trainable = False) prevents weights in these layers
        # from being updated during training.
        base_model.trainable = False

        fine_tune_at = cfg['fine_tune_at']

        # if fine tune at defined, freeze all the layers before the `fine_tune_at` layer
        if fine_tune_at > 0:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

        model = tf.keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(24, activation='softmax')
        ])

        return model, image_size, fine_tune_at



if __name__ == '__main__':

    mmaker = TransferModel()
    # build the basic model
    model = mmaker.build()
    model.summary()
