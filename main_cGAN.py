from keras.datasets import mnist, cifar10
from model.cGAN import cGAN
from model.funs import find_diff_label
from keras.optimizers import Adam

import keras
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

'''
0:airplane
1:automobile
2:bird
3:cat
4:deer
5:dog
6:frog
7:horse
8:ship
9:truck
'''

if __name__ == '__main__':

    # 設定每個隱向量的大小
    latent_size_tp = 200

    # 每隔幾步就輸出loss
    loss_output_size = 500

    # 設定訓練時的參數
    batch_size = 64
    epochs = 120

    # 宣告追蹤的數值之圖變化
    tracing_size = 49
    test_vector = tf.random.normal([tracing_size, latent_size_tp])
    test_label = np.random.randint(0, 9, size=tracing_size)
    test_label = tf.one_hot(test_label, 10)
    test_vector = tf.concat(
        [test_vector, test_label], axis=1
    )

    # 宣告儲存圖片的名字
    my_path = os.path.abspath(os.path.dirname(__file__))
    Images_name = my_path + '/photo_cGAN/photo_cifar_120/Images_Epochs_'

    # 宣告模型
    model = cGAN(opt_g=Adam(learning_rate=0.0002, beta_1=0.5), opt_d=Adam(learning_rate=0.0002, beta_1=0.5), latent_size=latent_size_tp, num_class=10, channels=3, width=32, height=32)

    # 藉由keras載入資料，並把資料做整理
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    concat_x = np.concatenate([X_train, X_test], axis = 0)
    concat_y = np.concatenate([y_train, y_test], axis = 0)

    # 將資料做正規化
    all_images = (concat_x.astype("float32") - 127.5) / 127.5
    all_images = np.reshape(all_images, (-1, 32, 32, 3))
    all_labels = keras.utils.to_categorical(concat_y, 10)
    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))

    # 使用tensorflow內建的dataset功能，幫助我們對資料進行隨機取樣
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    plt.figure(figsize=(12,12))

    # 開始訓練模型
    for e in range(epochs):

        print('\nEpochs %d' % (e+1))

        for itr, data in enumerate(dataset):
            
            d_loss, g_loss, generated_images = model.train_step(data, batch_size=batch_size, have_noise=False)

            if itr % loss_output_size == 0:
                print ('Itr: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (itr, d_loss, g_loss))
        
        trace_save_image = model.Generate_model.predict(test_vector)
        trace_save_image = np.squeeze(trace_save_image)

        for image_idx in range(tracing_size):
            plt.subplot(7, 7, image_idx+1) 
            plt.imshow(trace_save_image[image_idx], cmap='gray')
            plt.title("label %d" % (image_idx+1))

        plt.tight_layout()
        plt.savefig(Images_name+str(e+1))