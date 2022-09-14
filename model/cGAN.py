from keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout, GlobalMaxPooling2D
from keras.layers import LeakyReLU

import tensorflow as tf

class cGAN:
  def __init__(self , opt_g, opt_d, latent_size=100 ,width=28, height=28, num_class=10, channels=1):
    
    # 定義各種參數
    self._latent_size = latent_size
    self._width = width
    self._height = height
    self._channels = channels
    self._num_class = num_class

    # 輸入大小
    self.discriminator_in_channels = self._channels + self._num_class
    self.generator_in_channels = self._latent_size + self._num_class

    # 定義 loss function
    self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 定義優化器
    self.optimizer_g = opt_g
    self.optimizer_d = opt_d

    # 宣告生成網路
    self.Generate_model = self._bulit_generate()
    self.Generate_model.summary()

    # 宣告判別真假網路
    self.Discriminator_model = self._bulit_Discri()
    self.Discriminator_model.summary()

  def _bulit_generate(self):

    input = Input(shape=(self.generator_in_channels, ))
    d1 = Dense(7 * 7 * self.generator_in_channels)(input)
    d1 = LeakyReLU(alpha=0.2)(d1)
    d1_reshape = Reshape((7, 7, self.generator_in_channels))(d1)

    Conv1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(d1_reshape)
    Conv1 = LeakyReLU(alpha=0.2)(Conv1)

    Conv1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(Conv1)
    Conv1 = LeakyReLU(alpha=0.2)(Conv1)
    '''
    Conv1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(Conv1)
    Conv1 = LeakyReLU(alpha=0.2)(Conv1)
    '''
    out_image = Conv2D(1, (7, 7), padding="same", activation="sigmoid")(Conv1)

    generator = tf.keras.Model(inputs=input, outputs=out_image)

    return generator

  def _bulit_Discri(self):
    
    input = Input(shape=(28, 28, self.discriminator_in_channels))

    Conv1 = Conv2D(64, (3, 3), padding="same")(input)
    Conv1 = LeakyReLU(alpha=0.2)(Conv1)

    Conv1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(Conv1)
    Conv1 = LeakyReLU(alpha=0.2)(Conv1)

    Conv1 = GlobalMaxPooling2D()(Conv1)
    Flat = Flatten()(Conv1)
    out_image = Dense(1)(Flat)

    '''
    Conv1 = Conv2D(128, (3, 3), strides=(2, 2), padding="same")(Conv1)
    Conv1 = LeakyReLU(alpha=0.2)(Conv1)

    Conv1 = Conv2D(256, (3, 3), strides=(2, 2), padding="same")(Conv1)
    Conv1 = LeakyReLU(alpha=0.2)(Conv1)

    Flat = Flatten()(Conv1)
    Drop = Dropout(0.4)(Flat)
    out_image = Dense(1)(Drop)
    '''
    discrimintor = tf.keras.Model(inputs=input, outputs=out_image)

    return discrimintor

  @tf.function
  def train_step(self, data, batch_size=32, have_noise=True):

    # 導入實際資料
    real_images, one_hot_labels = data
    
    # 將one hot編碼的數據變成圖片維度
    image_one_hot_labels = one_hot_labels[:, :, None, None]
    image_one_hot_labels = tf.repeat(
        image_one_hot_labels, repeats=[self._width * self._height]
    )
    image_one_hot_labels = tf.reshape(
        image_one_hot_labels, (-1, self._width, self._height, self._num_class)
    )
    
    # 生成隨機的向量
    gen_noise = tf.random.normal([real_images.shape[0], self._latent_size])
    gen_noise_labels = tf.concat(
        [gen_noise, one_hot_labels], axis=1
    )

    generated_images = self.Generate_model(gen_noise_labels)

    # 把假的圖跟真的圖串在一起
    fake_combine_images = tf.concat([generated_images, image_one_hot_labels], -1)
    real_combine_images = tf.concat([real_images, image_one_hot_labels], -1)
    combined_images = tf.concat([
        fake_combine_images, real_combine_images], axis = 0
        )

    y_combined_data = tf.concat([tf.ones([real_images.shape[0], 1]), tf.zeros([real_images.shape[0], 1])], 0)
    
    # 官網說這是一個很重要的方式?!(maybe很好收斂)
    if have_noise:
        y_combined_data += 0.05 * tf.random.uniform(y_combined_data.shape)

    predict_labes = self.Discriminator_model(combined_images)

    # 訓練discrimintor model
    with tf.GradientTape() as tape:
        predict_labes = self.Discriminator_model(combined_images)
        d_loss = self.loss_fn(y_combined_data, predict_labes)
    grads = tape.gradient(d_loss, self.Discriminator_model.trainable_weights)
    self.optimizer_d.apply_gradients(zip(grads, self.Discriminator_model.trainable_weights))

    # 生成隨機的向量
    gen_noise = tf.random.normal([real_images.shape[0], self._latent_size])
    gen_noise_labels = tf.concat(
        [gen_noise, one_hot_labels], axis=1
    )
    y_mislabels_data = tf.zeros([real_images.shape[0], 1])

    with tf.GradientTape() as tape:
        fake_images = self.Generate_model(gen_noise_labels)
        fake_image_labels = tf.concat(
            [fake_images, image_one_hot_labels], -1
        )
        predict_labes = self.Discriminator_model(fake_image_labels)
        g_loss = self.loss_fn(y_mislabels_data, predict_labes)
    grads = tape.gradient(g_loss, self.Generate_model.trainable_weights)
    self.optimizer_g.apply_gradients(zip(grads, self.Generate_model.trainable_weights))

    return d_loss, g_loss, generated_images
