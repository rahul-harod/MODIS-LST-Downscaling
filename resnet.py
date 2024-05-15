from keras.layers import Dropout  # Import Dropout layer
from keras.layers import Input, Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, AveragePooling2D,AveragePooling3D, Flatten, Dense, Add
from keras.models import Model
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Reshape
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

Dropout_Rate=0.2

def identity_block(input_tensor, kernel_size, filters, stage, block, dropout_rate=Dropout_Rate):
    """The identity block is the block that has no conv layer at shortcut."""
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = f'res{stage}_{block}_branch'
    bn_name_base = f'bn{stage}_{block}_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)  # Add dropout here

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)  # Add dropout here

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1), dropout_rate=Dropout_Rate):
    """A block that has a conv layer at shortcut."""
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = f'res{stage}_{block}_branch'
    bn_name_base = f'bn{stage}_{block}_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)  # Add dropout here

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)  # Add dropout here

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(num_rows, num_Columns, bands):
    """Instantiates the ResNet50 architecture."""
    bn_axis = -1

    input_tensor = Input(shape=(num_rows, num_Columns, bands))
    x=input_tensor
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='d')

    x = conv_block(x, 3, [64, 64, 256], stage=4, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='d')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='e')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='f')

    x = conv_block(x, 3, [64, 64, 32], stage=5, block='a')
    x = identity_block(x, 3, [64, 64, 32], stage=5, block='b')
    x = identity_block(x, 3, [64, 64, 32], stage=5, block='c')
    print('X:',x.shape)
    
    x = Reshape((-1, 32))(x)
    print('X:',x.shape)
    x = Dense(10, activation='relu', name='fc1')(x)  # Additional dense layer
    x = Dropout(Dropout_Rate)(x)  # Dropout layer for regularization
    x = Dense(1, activation='linear', name='fc2')(x)  # Use linear activation for regression
    
    x = Reshape((num_rows, num_Columns))(x)
    print('X_reshape',x.shape)
    
    model = Model(inputs=input_tensor, outputs=x, name='resnet50')
    return model
