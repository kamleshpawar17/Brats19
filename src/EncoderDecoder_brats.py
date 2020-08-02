from keras.applications.densenet import DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add, Cropping2D, \
    ZeroPadding2D, Input, MaxPooling2D, SeparableConv2D, UpSampling2D, Lambda, Conv3D, SpatialDropout3D, \
    UpSampling3D, Cropping3D, ZeroPadding3D, SpatialDropout2D
from keras.models import Model
import keras.backend as K
import keras_applications
import keras


def conv2D_BN_Act(nFeature, kSize, kStride, inp, padding='same', sdo=False, sdo_frac=0.2):
    C = Conv2D(nFeature, kernel_size=kSize, strides=kStride, padding=padding)(inp)
    C_BN = BatchNormalization()(C)
    if sdo:
        C_BN = SpatialDropout2D(rate=sdo_frac)(C_BN)
    C_BN_Act = Activation('relu')(C_BN)
    return C_BN_Act


def UpSampling2D_BN_Act(kSize, crop, outPad, concateInp, inp):
    C = UpSampling2D(size=kSize)(inp)
    C_Crop = Cropping2D(cropping=crop)(C)
    C_Zpad = ZeroPadding2D(padding=outPad)(C_Crop)
    C_Con = concatenate([C_Zpad, concateInp], axis=-1)
    return C_Con


def append_inputchannels(model_inp, H, W, ninchannel):
    inp = Input(shape=(H, W, ninchannel))
    C1 = conv2D_BN_Act(nFeature=3, kSize=3, kStride=1, inp=inp, padding='same')
    out = model_inp(C1)
    model_out = Model(inputs=[inp], outputs=[out])
    return model_out


def ResNet50_enc_dec(H=256, W=256, weights='imagenet', ninchannel=3, noutchannel=1, isregression=True,
                     ismuticlass=False, sdo=False, sdo_frac=0.2):
    """
    This is an encoder-decoder network architecture for image reconstruction and image segmentation
    based on pretrained ResNet50 model as encoder. The deocder need to be trained

    :param img_sz: size of the input image [n x n x 3]
    :param noutchannel: The number of output channels of the network, only valid if ismuticlass==True else ignored
    :param isregression: If ==True then model output is single channel without any activation applied to the last layer, ismuticlass and noutchannel is ignored
                         If ==False and ismuticlass==False, then output is single channel and sigmoid activation applied to the last layer
                         If ==False and ismuticlass==True, then output is multichannel determined by noutchannel and softmax activation is appled to the last layer
    :param ismuticlass:  If ==False then output is single channel and sigmoid activation applied to the last layer
                         If ==True then output is multichannel determined by noutchannel and softmax activation is appled to the last layer
    :return: Encoder Decoder Model
    """
    model = ResNet50(include_top=False, weights=weights, input_tensor=None, input_shape=(H, W, 3), pooling=None)

    # ----- Decoder Network ------ #
    nFeat_d = 256
    convT_0_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=model.get_layer('activation_40').output,
                                      inp=model.get_layer('activation_49').output)
    conv_0_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_0_0_d, sdo=sdo, sdo_frac=sdo_frac)
    conv_0_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_0_0_d, sdo=sdo, sdo_frac=sdo_frac)

    convT_1_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=model.get_layer('activation_22').output,
                                      inp=conv_0_1_d)
    conv_1_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_1_0_d, sdo=sdo, sdo_frac=sdo_frac)
    conv_1_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_1_0_d, sdo=sdo, sdo_frac=sdo_frac)

    convT_2_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=model.get_layer('activation_10').output,
                                      inp=conv_1_1_d)
    conv_2_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_2_0_d, sdo=sdo, sdo_frac=sdo_frac)
    conv_2_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_2_0_d, sdo=sdo, sdo_frac=sdo_frac)

    convT_3_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=model.get_layer('activation_1').output,
                                      inp=conv_2_1_d)
    conv_3_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_3_0_d, sdo=sdo, sdo_frac=sdo_frac)
    conv_3_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_3_0_d, sdo=sdo, sdo_frac=sdo_frac)

    convT_4_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=model.input, inp=conv_3_1_d)
    conv_4_0_d = conv2D_BN_Act(nFeature=nFeat_d / 2, kSize=3, kStride=1, inp=convT_4_0_d, sdo=sdo, sdo_frac=sdo_frac)

    if (isregression):
        conv_4_1_d = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(conv_4_0_d)
    else:
        if (ismuticlass):
            conv_4_1_d = Conv2D(filters=noutchannel, kernel_size=3, strides=1, padding='same')(conv_4_0_d)
            conv_4_1_d = Activation('softmax')(conv_4_1_d)
        else:
            conv_4_1_d = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(conv_4_0_d)
            conv_4_1_d = Activation('sigmoid')(conv_4_1_d)

    model_1 = Model(inputs=[model.input], outputs=[conv_4_1_d])
    if ninchannel != 3:
        model_1 = append_inputchannels(model_1, H, W, ninchannel)
    return model_1


def NASNetMobile_enc_dec(H=256, W=256, weights='imagenet', ninchannel=3, noutchannel=1, isregression=True,
                         ismuticlass=False, sdo=False, sdo_frac=0.2):
    """
    This is an encoder-decoder network architecture for image reconstruction and image segmentation
    based on pretrained NASNetLarge model as encoder. The deocder need to be trained

    :param img_sz: size of the input image [n x n x 3]
    :param noutchannel: The number of output channels of the network, only valid if ismuticlass==True else ignored
    :param isregression: If ==True then model output is single channel without any activation applied to the last layer, ismuticlass and noutchannel is ignored
                         If ==False and ismuticlass==False, then output is single channel and sigmoid activation applied to the last layer
                         If ==False and ismuticlass==True, then output is multichannel determined by noutchannel and softmax activation is appled to the last layer
    :param ismuticlass:  If ==False then output is single channel and sigmoid activation applied to the last layer
                         If ==True then output is multichannel determined by noutchannel and softmax activation is appled to the last layer
    :return: Encoder Decoder Model
    """
    model = NASNetMobile(include_top=False, weights=weights, input_tensor=None, input_shape=(H, W, 3),
                         pooling=None)
    # ----- Decoder Network ------ #
    nFeat_d = 256
    convT_0_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0,
                                      concateInp=add([model.get_layer('activation_129').output,
                                                      model.get_layer('activation_130').output]),
                                      inp=model.get_layer('activation_188').output)
    conv_0_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_0_0_d, sdo=sdo, sdo_frac=sdo_frac)
    conv_0_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_0_0_d, sdo=sdo, sdo_frac=sdo_frac)

    convT_1_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0,
                                      concateInp=add([model.get_layer('activation_70').output,
                                                      model.get_layer('activation_71').output]),
                                      inp=conv_0_1_d)
    conv_1_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_1_0_d, sdo=sdo, sdo_frac=sdo_frac)
    conv_1_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_1_0_d, sdo=sdo, sdo_frac=sdo_frac)

    convT_2_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0,
                                      concateInp=add([model.get_layer('activation_13').output,
                                                      model.get_layer('activation_15').output,
                                                      model.get_layer('activation_17').output,
                                                      model.get_layer('activation_19').output]),
                                      inp=conv_1_1_d)
    conv_2_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_2_0_d, sdo=sdo, sdo_frac=sdo_frac)
    conv_2_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_2_0_d, sdo=sdo, sdo_frac=sdo_frac)

    convT_3_0_d = UpSampling2D_BN_Act(kSize=2, crop=((0, 1), (0, 1)), outPad=0,
                                      concateInp=add([model.get_layer('activation_1').output,
                                                      model.get_layer('activation_4').output,
                                                      model.get_layer('activation_6').output,
                                                      model.get_layer('activation_8').output]),
                                      inp=conv_2_1_d)
    conv_3_0_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=convT_3_0_d, sdo=sdo, sdo_frac=sdo_frac)
    conv_3_1_d = conv2D_BN_Act(nFeature=nFeat_d, kSize=3, kStride=1, inp=conv_3_0_d, sdo=sdo, sdo_frac=sdo_frac)

    convT_4_0_d = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=1,
                                      concateInp=model.input,
                                      inp=conv_3_1_d)
    conv_4_0_d = conv2D_BN_Act(nFeature=nFeat_d / 2, kSize=3, kStride=1, inp=convT_4_0_d, sdo=sdo, sdo_frac=sdo_frac)

    if (isregression):
        conv_4_1_d = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(conv_4_0_d)
    else:
        if (ismuticlass):
            conv_4_1_d = Conv2D(filters=noutchannel, kernel_size=3, strides=1, padding='same')(conv_4_0_d)
            conv_4_1_d = Activation('softmax')(conv_4_1_d)
        else:
            conv_4_1_d = Conv2D(filters=1, kernel_size=3, strides=1, padding='same')(conv_4_0_d)
            conv_4_1_d = Activation('sigmoid')(conv_4_1_d)

    model_1 = Model(inputs=[model.input], outputs=[conv_4_1_d])
    if ninchannel != 3:
        model_1 = append_inputchannels(model_1, H, W, ninchannel)
    return model_1
