SOURCE = 'https://bitbucket.org/mbari/mlflow-kerasclassifier/src/master/train.py'

MODEL_DICT = {}

MIN_LR = 0.5#1e-2
MAX_LR = 0.1#1e-1

densenet201 = dict(
    image_size = 224,
    model_instance = "keras.applications.densenet.DenseNet201",
    fine_tune_at = -1,
    has_depthwise_layers = False
)
inceptionv3 = dict(
    image_size = 299,
    model_instance = "keras.applications.inception_v3.InceptionV3",
    fine_tune_at = -1,
    has_depthwise_layers = False
)
inception_resnetv2 = dict(
    image_size = 299,
    model_instance = "keras.applications.inception_resnet_v2.InceptionResNetV2",
    fine_tune_at = -1,
    has_depthwise_layers = False
)
xception = dict(
    image_size = 299,
    model_instance="keras.applications.xception.Xception",
    fine_tune_at = -1,
    has_depthwise_layers = False
)
nasnetlarge = dict(
    image_size = 331,
    model_instance="keras.applications.nasnet.NASNetLarge",
    fine_tune_at = -1,
    has_depthwise_layers = False
)
resnet50 = dict(
    image_size = 224,
    model_instance="keras.applications.ResNet50",
    fine_tune_at = -1,
    has_depthwise_layers = False
)
vgg16 = dict(
    image_size = 150,
    model_instance="keras.applications.VGG16",
    fine_tune_at = -1,
    has_depthwise_layers = False
)
mobilenetv2 = dict(
    image_size = 224,
    model_instance = "keras.applications.MobileNetV2",
    model_url  = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2",
    feature_extractor_url="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2",
    fine_tune_at = 100,
    has_depthwise_layers = True
)
MODEL_DICT["densenet201"] = densenet201
MODEL_DICT["xception"] = xception
MODEL_DICT["inceptionv3"] = inceptionv3
MODEL_DICT["inception_resnetv2"] = inception_resnetv2
MODEL_DICT["nasnetlarge"] = nasnetlarge
MODEL_DICT["resnet50"] = resnet50
MODEL_DICT["mobilenetv2"] = mobilenetv2
MODEL_DICT["vgg16"] = vgg16
