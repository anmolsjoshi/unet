from tensorflow.python import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, UpSampling2D, Cropping2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.losses import binary_crossentropy


class UNet:
    
    def __init__(self, img_dim=None):
        self.img_shape = (img_dim, img_dim, 3)
    
    def Conv2D_x2_Pool(self, x, filters, pool=True):
        conv = Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(x)
        res = Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv)
        if pool:
            out = MaxPooling2D()(res)
            return out, res
        else:
            return res
    
    def UpConv(self, x, res, filters):
        x = UpSampling2D()(x)
        conv = Conv2D(filters=filters, kernel_size=(2,2), padding='same')(x)
        cropping_size = res.get_shape().as_list()[1] - conv.get_shape().as_list()[1]
        crop = Cropping2D(cropping=cropping_size//2)(res)
        merged = Concatenate()([conv, crop])
        conv_op_1 = Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu')(merged)
        out = Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation='relu')(conv_op_1)    
        return out
    
    def create_model(self):
        img = Input(shape=self.img_shape)
        filters = [64, 128, 256, 512, 1024]

        x = img
        residuals = []
        pool = True

        for fil in filters:

            if fil == 1024:
                pool = False

            if pool == True: 
                x, res = self.Conv2D_x2_Pool(x, fil, pool=pool)
                residuals.append(res)

            else:
                x = self.Conv2D_x2_Pool(x, fil, pool=pool)
        
        counter = -1
        
        for fil in reversed(filters[:-1]):

            x = self.UpConv(x, residuals[counter], fil) 
            
            counter = counter - 1
            
        out = Conv2D(filters=1, kernel_size=(1,1), padding='same', activation='sigmoid')(x)
        
        model = Model(inputs=img, outputs=out)
        
        return model

def dice_coeff(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def dice_coeff_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)



img_size = 256
image = 'images'
masks = 'masks'

seed = 1
DATA_DIR = '../../data/raw/car/'

data_gen_args = dict(rescale=1./255)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_args = dict(seed=seed,
    batch_size=17,
    shuffle=True,
    class_mode=None,
    target_size=(img_size, img_size),
    color_mode='rgb')

mask_args = dict(seed=seed,
    batch_size=17,
    class_mode=None,
    shuffle=True,
    target_size=(img_size, img_size), 
    color_mode='grayscale')
test_generator = zip(image_datagen.flow_from_directory(**image_args, directory=DATA_DIR+'test_'+image),  
                      mask_datagen.flow_from_directory(**mask_args, directory=DATA_DIR+'test_'+masks))


net = UNet(img_dim=img_size)
model = net.create_model()
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy',
                  metrics=[dice_coeff])
filename = 'weights/Adam_BCE_1e-4_best_weights.hdf5'
parallel_model.load_weights(filename)


print (parallel_model.evaluate_generator(test_generator, steps=15))
