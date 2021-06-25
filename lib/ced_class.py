import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Reshape, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Softmax
from tensorflow.keras.layers import BatchNormalization, Activation, ReLU, LeakyReLU
from tensorflow.keras.models import Model

class Attention(keras.layers.Layer):
    def __init__(self, filters):
        super(Attention, self).__init__()
        
        self.query_conv = Conv2D(filters, kernel_size=1, padding='same')
        self.key_conv = Conv2D(filters, kernel_size=1, padding='same')
        self.value_conv = Conv2D(filters, kernel_size=1, padding='same')
        
        self.gamma = K.zeros(shape=(1,))
    
    def call(self, x):
        # get the size
        batchsize, width, height, c = x.shape
        
        # original
        q = self.query_conv(x)
        k = self.key_conv(x)
        v = self.value_conv(x)
        
        # reshape
        qt = K.reshape(q, [-1, width*height, c])
        kt = K.reshape(k, [-1, width*height, c])
        vt = K.reshape(v, [-1, width*height, c])
        
        s = K.batch_dot(K.permute_dimensions(qt, [0, 2, 1]), kt)
        scores = K.batch_dot(vt, K.softmax(s))
        scores = K.reshape(scores, [-1, width, height, c])
        
        return x + self.gamma * scores

class Enc_block(tensorflow.keras.layers.Layer):
    def __init__(self, filters, kernel_size, momentum, beta, downsampling=False, num=1):
        super(Enc_block, self).__init__()
        
        # parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.momentum = momentum
        self.beta = beta
        
        # number of blocks
        self.num = num
        self.layer_list = []
        
        # for the first conv2d-BN-LeakyReLU block, need to check if downsampling is needed
        # if downsampling, first conv2d layer use (2, 2) for the strides
        if downsampling:
            self.layer_list.append(Conv2D(filters=filters, kernel_size=kernel_size,
                                              strides=(2, 2), padding='same'))
        else:
            self.layer_list.append(Conv2D(filters=filters, kernel_size=kernel_size,
                                              strides=(1, 1), padding='same'))
        self.layer_list.append(BatchNormalization(momentum=momentum))
        self.layer_list.append(LeakyReLU(beta))
        
        # rest of the part will be the same
        for i in range(self.num-1):
            self.layer_list.append(Conv2D(filters=filters, kernel_size=kernel_size,
                                          strides=(1, 1), padding='same'))
            self.layer_list.append(BatchNormalization(momentum=momentum))
            self.layer_list.append(LeakyReLU(beta))
    
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

class Encoder(tensorflow.keras.Model):
    def __init__(self, img_shape, filters, kernel_size,
                       momentum=0.8, beta=0.3, num=1, attention=False, rate=0.2):
        super(Encoder, self).__init__()
        
        self.attention = attention
        self.rate = rate
        
        self.eb1 = Enc_block(filters,    kernel_size[0], momentum, beta, False, num)
        self.eb2 = Enc_block(filters*2,  kernel_size[0], momentum, beta, True,  num)
        self.eb3 = Enc_block(filters*4,  kernel_size[0], momentum, beta, True,  num)
        self.eb4 = Enc_block(filters*8,  kernel_size[1], momentum, beta, True,  num)
        self.eb5 = Enc_block(filters*16, kernel_size[1], momentum, beta, True,  num)
        self.eb6 = Enc_block(filters*32, kernel_size[1], momentum, beta, True,  num)
        
        if self.attention:
            self.sa1 = Attention(filters*8)
            self.sa2 = Attention(filters*16)
    
    def call(self, x):
        e1 = self.eb1(x)
        e2 = self.eb2(e1)
        e3 = self.eb3(e2)
        e4 = self.eb4(e3)
        if self.attention:
            e4 = self.sa1(e4)
        e5 = self.eb5(e4)
        if self.attention:
            e5 = self.sa2(e5)
        e6 = self.eb6(e5)
        
        e_out = Flatten()(e6)
        e_out = Dropout(self.rate)(e_out)
        e_out = Reshape((1, 1, 2048))(e_out)
        
        return e_out

class Dec_block(tensorflow.keras.layers.Layer):
    def __init__(self, filters, kernel_size, momentum, beta, upsampling=False, num=1):
        super(Dec_block, self).__init__()
        
        # parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.momentum = momentum
        self.beta = beta
        
        # number of blocks, each block contains one conv2d, one BN, and one LeakyReLU
        self.num = num
        self.layer_list = []
        
        # for the first conv2d-BN-LeakyReLU block, check using upsampling or conv2d transpose
        # if upsampling, do upsampling followed by a conv2d layer
        # if not, use conv2d transpose with strides 2
        if upsampling:
            self.layer_list.append(UpSampling2D(interpolation="bilinear"))
            self.layer_list.append(Conv2D(filters=filters, kernel_size=kernel_size,
                                          strides=(1, 1), padding='same'))
        else:
            self.layer_list.append(Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                   strides=(2, 2), padding='same'))
        self.layer_list.append(BatchNormalization(momentum=momentum))
        self.layer_list.append(LeakyReLU(beta))
        
        # rest of the part will be the same
        for i in range(self.num-1):
            self.layer_list.append(Conv2D(filters=filters, kernel_size=kernel_size,
                                          strides=(1, 1), padding='same'))
            self.layer_list.append(BatchNormalization(momentum=momentum))
            self.layer_list.append(LeakyReLU(beta))
    
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

class Decoder(tensorflow.keras.Model):
    def __init__(self, img_shape, filters, kernel_size,
                       momentum=0.8, beta=0.3, num=1, attention=False, rate=0.2):
        super(Decoder, self).__init__()
        
        # parameters
        self.filters = filters
        self.kernel_size = kernel_size
        self.momentum = momentum
        self.beta = beta
        self.attention = attention
        self.rate = rate
        
        self.db1 = Dec_block(filters*32, kernel_size[0], momentum, beta, False, num)
        self.db2 = Dec_block(filters*16, kernel_size[0], momentum, beta, False, num)
        self.db3 = Dec_block(filters*8,  kernel_size[0], momentum, beta, False, num)
        self.db4 = Dec_block(filters*4,  kernel_size[1], momentum, beta, False, num)
        self.db5 = Dec_block(filters*2,  kernel_size[1], momentum, beta, False, num)
        self.db6 = Dec_block(filters,    kernel_size[1], momentum, beta, False, num)
        
        if self.attention:
            self.sa1 = Attention(filters*4)
            self.sa2 = Attention(filters*2)
    
    def call(self, x):
        d1 = self.db1(x)
        d2 = self.db2(d1)
        d3 = self.db3(d2)
        d4 = self.db4(d3)
        if self.attention:
            d4 = self.sa1(d4)
        d5 = self.db5(d4)
        if self.attention:
            d5 = self.sa2(d5)
        d6 = self.db6(d5)
        
        d_out = Conv2D(filters=1, kernel_size=self.kernel_size[1], strides=(1, 1), padding='same')(d6)
        d_out = BatchNormalization(momentum=self.momentum)(d_out)
        d_out = ReLU()(d_out)
        d_out = Dropout(self.rate)(d_out)
        
        return d_out

def build_ced(img_shape, base=16, attention=False):
    """ Create encoder and decoder """
    # encoder = build_encoder(img_shape=img_shape,
                            # kernels=[(5, 5), (3, 3)],
                            # base=base,
                            # attention=attention)
    encoder = Encoder(img_shape, filters=base, kernel_size=[(5, 5), (3, 3)],
                      num=3, attention=attention, rate=0.4)
    input_img = Input(shape=img_shape)
    encodings = encoder(input_img)
    
    # input shape of decoder is the same as output shape of encoder
    dec_input_shape = encodings.shape[1:]
    #intermediate = build_intermediate(input_shape=dec_input_shape)
    decoder = Decoder(dec_input_shape, filters=base, kernel_size=[(3, 3), (3, 3)],
                      num=3, attention=attention, rate=0.4)

    # Build CED (combined model)
    #medium = intermediate(encodings)
    recon_img = decoder(encodings)
    model = Model(inputs=input_img, outputs=recon_img)
    
    return model

def sample_images(model, x_test, y_test, epoch, path, num_samples=5):
    if not os.path.isdir(path):
        os.mkdir(path)
    
    r, c = 2, num_samples
    idx = np.random.choice(len(x_test), size=c, replace=False)
    idx = np.arange(len(x_test))

    recons = model.predict(x_test[idx])
    
    fig, axs = plt.subplots(r, c, figsize=(5*num_samples, 10))
    for j in range(c):
        axs[0,j].imshow(recons[j,:,:,0], cmap='gray')
        axs[0,j].set_title("Recon {}".format(j))
        axs[0,j].axis('off')
        
        axs[1,j].imshow(y_test[idx[j],:,:,0], cmap='gray')
        axs[1,j].set_title("Real {}".format(j))
        axs[1,j].axis('off')
    
    fig.savefig("{}result_iter{:06}.png".format(path, epoch))
    plt.close()
    return
