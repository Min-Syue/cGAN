from keras.datasets import mnist, cifar10
from model.cGAN import cGAN
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
    # Mnist latent size 128
    # Cifar latent size 100
    latent_size_tp = 100

    # 每隔幾步就輸出loss
    loss_output_size = 500

    # 設定訓練時的參數
    batch_size = 64
    epochs = 100

    # 宣告追蹤的數值之圖變化
    tracing_size = 49
    test_vector = tf.random.normal([tracing_size, latent_size_tp])
    test_label_np = np.random.randint(0, 9, size=tracing_size)
    test_label = tf.one_hot(test_label_np, 10)
    test_vector = tf.concat(
        [test_vector, test_label], axis=1
    )

    # 宣告儲存圖片的名字
    my_path = os.path.abspath(os.path.dirname(__file__))
    # 儲存路徑 photo_mnist 或 photo_cifar
    Images_name = my_path + '/photo_cGAN/photo_cifar/Images_Epochs_'

    # 宣告模型
    # Mnist 圖片大小為 28 x 28 x 1
    # Cifar 圖片大小為 32 x 32 x 3
    # 兩者皆為 Multiclass classification 且皆為10個分類目標， 因此 num_class = 10
    model = cGAN(opt_g=Adam(learning_rate=0.0002, beta_1=0.5), opt_d=Adam(learning_rate=0.0002, beta_1=0.5)
                 , latent_size=latent_size_tp, num_class=10, channels=3, width=32, height=32, DataName='cifar10')

    # 藉由keras載入資料，並把資料做整理
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    concat_x = np.concatenate([X_train, X_test], axis = 0)
    concat_y = np.concatenate([y_train, y_test], axis = 0)

    # 將資料做正規化
    # Mnist 的 Normalization 
    # all_images = (concat_x.astype("float32")) / 255.0
    # all_images = np.reshape(all_images, (-1, 28, 28, 1))

    # Cifar 的 Normalization 
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
            
            d_loss, g_loss = model.train_step(data)

            if itr % loss_output_size == 0:
                print ('Itr: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (itr, d_loss, g_loss))
        
        trace_save_image = model.Generate_model.predict(test_vector)
        trace_save_image = np.squeeze(trace_save_image)

        for image_idx in range(tracing_size):
            plt.subplot(7, 7, image_idx+1) 
            plt.imshow(trace_save_image[image_idx], cmap='gray')
            plt.title("label %d" % (test_label_np[image_idx]))

        plt.tight_layout()
        plt.savefig(Images_name+str(e+1))