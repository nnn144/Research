import os
import sys
import cv2
import lib.ced_class as ced2
import time
import tensorflow
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def train():
    # load training data
    path1 = "../../../../store01/groups/scw1489/Haoran/mouse_phantom/data/x_train_phantom81.npy"
    path2 = "../../../../store01/groups/scw1489/Haoran/mouse_phantom/data/y_train_phantom81.npy"
    x_train = np.load(path1)
    y_train = np.load(path2)    
    #x_train, y_train = data_generation(x_train, y_train, 1)
    theta = np.linspace(0., 180., 64, endpoint=True)
    x_train2 = np.zeros((x_train.shape))
    for i in range(x_train.shape[0]):
        x_train2[i] = iradon(x_train[i], theta=theta)
    
    # load testing data
    path1 = "../../../../store01/groups/scw1489/Haoran/mouse_phantom/data/total_sinos_cropped.npy"
    path2 = "../../../../store01/groups/scw1489/Haoran/mouse_phantom/data/total_images_cropped.npy"
    x_test = np.load(path1)
    y_test = np.load(path2)
    x_test2 = np.zeros((y_test.shape))
    for i in range(y_test.shape[0]):
        x_test2[i] = iradon(y_test[i], theta=theta)

    # add new axis
    x_train = normalize_individual(x_train)
    x_train2 = normalize_individual(x_train2)
    y_train = normalize_individual(y_train)
    x_train = x_train[..., np.newaxis]
    x_train2 = x_train2[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    print (x_train.shape, y_train.shape)
    
    x_test = normalize_individual(x_test)
    x_test2 = normalize_individual(x_test2)
    y_test = normalize_individual(y_test)
    x_test = x_test[..., np.newaxis]
    x_test2 = x_test2[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    print (x_test.shape, y_test.shape)

    # shuffle the data order
    seed = 123
    order = np.arange(len(x_train))
    np.random.seed(seed)
    np.random.shuffle(order)
    x_train = x_train[order]
    x_train2 = x_train2[order]
    y_train = y_train[order]

    # get the arguments
    #img_rows, img_cols, img_channels = y_train[0].shape
    #img_shape = (img_rows, img_cols, img_channels)
    input_shape = x_train[0].shape
    
    #
    #blur_img_set = blur_img_set[idx]

    print ("Training set: {} images, testing set: {} images.".format(len(x_train), len(x_test)))
    #print ("We are trying to reconstruct the image from the blurred sinogram.")


    # build the model and train using multiple gpu
    model = ced2.build_ced(img_shape=input_shape, flag=1, attention=True)
    #model = multi_gpu_model(model, gpus=2)
    
    """
    encoder = ced2.build_encoder(img_shape=input_shape, kernels=[(5, 5), (3, 3)], base=16)
    encoder2 = ced2.build_encoder(img_shape=input_shape, kernels=[(5, 5), (3, 3)], base=16)
    dec_input_shape = encoder.layers[-1].output_shape[1:]
    decoder = ced2.build_decoder(input_shape=dec_input_shape, kernels=(3, 3), base=16, flag=1)

    # Build CED (combined model)
    input_img = Input(shape=input_shape)
    encodings = encoder(input_img)
    
    input_img2 = Input(shape=input_shape)
    encodings2 = encoder2(input_img2)
    
    #medium = intermediate(encodings)
    recon_img = decoder(encodings+encodings2)
    model = Model(inputs=[input_img, input_img2], outputs=recon_img)
    """
    
    # compile
    adam = Adam(0.005)
    model.compile(optimizer=adam, loss='mse')
    #model.compile(optimizer=adam, loss=ssim_loss)

    # start training
    epochs = 100
    batch_size = 25
    steps = len(x_train) // batch_size
    tl_list = []
    el_list = []
    
    #current_time = time.localtime()
    #dir_name = time.strftime("%y%m%d-%H%M%S", current_time)
    #dir_name = "gate" + str(num) + "_" + "83-123_" + dir_name
    dir_path = os.path.join("./ced_result/", "phantom_pretrain3")
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    
    report = open("./report/report_phantom_pretrain3.txt", 'w')
    report.write("Start training:\n")
    # start counting time
    start = time.time()
    for i in range(epochs):
        for j in range(steps):
            s = j * batch_size
            e = (j + 1) * batch_size
            train_loss = model.train_on_batch(x_train[s:e, ...], y_train[s:e, ...])
            #train_loss = model.train_on_batch([x_train[s:e, ...], x_train2[s:e, ...]], y_train[s:e, ...])
            train_loss = model.train_on_batch(x_train[s:e, ...], y_train[s:e, ...])
        
        # evaluation loss
        evaluate_loss = model.evaluate(x_test, y_test, verbose=0)
        #evaluate_loss = model.evaluate([x_test, x_test2], y_test, verbose=0)
        tl_list.append(train_loss)
        el_list.append(evaluate_loss)
        
        # save the figures
        ced2.sample_images(model, x_test, y_test, i, dir_path+"/")
        #ced2.sample_images(model, [x_test, x_test2], y_test, i, dir_path+"/")
        
        # training log
        iter_num = "Epoch {:03}".format(i)
        train_loss_info = "Training loss is {:.8f}".format(train_loss)
        evaluate_loss_info = "Evaluation loss is {:.8f}".format(evaluate_loss)
        log_info = "{} -- {}. {}\n".format(iter_num,
                                           train_loss_info,
                                           evaluate_loss_info)
        # print out the log info
        print (log_info)
        report.write(log_info)
    # end counting time
    end = time.time()
    report.write("Training End.\n")
    report.write("============================================================================\n")
    report.write("Training time: {} seconds.\n".format(end-start))
    start = time.time()
    result = model.evaluate([x_test, x_test2], y_test, verbose=0)
    end = time.time()
    report.write("Evaluating time: {} seconds. Average recon time for single case: {} seconds.\n".format
           (end-start, (end-start)/len(x_test)))
    report.write("Evalutaion error: {}\n".format(result))
    report.close()
    
    # save model
    #model.save("phantom_pretrain2.h5")
    
if __name__ == "__main__":
    train()
