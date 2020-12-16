import tensorflow as tf
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Conv2D,Conv2DTranspose, Dense, Flatten, Reshape, InputLayer,Layer,MaxPool2D
from tensorflow.keras.layers import InputSpec, Input, Dense,BatchNormalization,Activation,GlobalAveragePooling2D,Concatenate,Add
import numpy as np
import os

class CAE_MNIST_VGG(tf.keras.models.Model):
    def __init__(self,embedding_size,input_size=(28,28,1),if_batch_norm=False,
                 if_extra_dense=False,**kwargs):
        super(CAE_MNIST_VGG, self).__init__(**kwargs)
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.if_batch_norm = if_batch_norm
        self.if_extra_dense = if_extra_dense
        self.encoder = self.enocder_fun()
        self.decoder = self.decoder_fun()
        
    def enocder_fun(self):
        input_ = Input(shape=self.input_size)
        vgg_layer1 = self.vgg_block(input_,32,2)
        vgg_layer2 = self.vgg_block(vgg_layer1,64,2)
        vgg_layer3 = self.vgg_block(vgg_layer2,128,2)
        flattening = Flatten()(vgg_layer3)
        if self.if_extra_dense:
            extra_dense1 = Dense(4*self.embedding_size,activation="relu",name="extra_dense1")(flattening)
            extra_dense2 = Dense(2*self.embedding_size,activation="relu",name="extra_dense2")(extra_dense1)
            
            out = Dense(self.embedding_size,activation="relu")(extra_dense2)
            self.units = flattening.shape[1]
            self.last_con_shape = vgg_layer3.shape
        else:
            out = Dense(self.embedding_size,activation="relu")(flattening)
            self.units = flattening.shape[1]
            self.last_con_shape = vgg_layer3.shape
        encoder = Model(inputs=input_,outputs=out,name="encoder")
        return encoder
    
    def decoder_fun(self):
        input_ = Input(shape = self.embedding_size)
        x = Dense(self.units,activation="relu")(input_)
        x = Reshape(self.last_con_shape[1:])(x)
        x = Conv2DTranspose(128,3,strides=2)(x)
        if self.if_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(64,3,strides=2,padding="same")(x)
        if self.if_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(32,3,strides=2,padding="same")(x)
        if self.if_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(1,3,strides=1,padding="same")(x)
        output = Activation('sigmoid')(x)
        decoder = Model(inputs=input_,outputs=output,name="decoder")
        return decoder
   
    
    def vgg_block(self,input_layer,no_of_filters,no_of_conv):
        for _ in range(no_of_conv):
            input_layer = Conv2D(no_of_filters,(3,3), padding='same')(input_layer)
            if self.if_batch_norm:
                input_layer = BatchNormalization()(input_layer)
            input_layer = Activation('relu')(input_layer)
        input_layer = MaxPool2D((2,2), strides=(2,2))(input_layer)
        return input_layer