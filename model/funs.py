import numpy as np
import imageio
import os

def find_diff_label(Train_data, label_data, label = 0):

    # 從label中尋找每個label的index
    idx_train = np.where(label_data == label)
    # 把上一步找到的index，放到Train_data尋找每個訓練資料
    output_train = Train_data[idx_train[0]]

    return output_train

def make_gif(input_adr, output_name, interval = 3, FPS = 0.8, top_range=151):

    file_names = []
    images = []

    my_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    Images_name = my_path + input_adr 

    file_names.append(Images_name + str(1) + '.png')

    for i in range(2, top_range):
        if i % interval == 0:
            file_names.append(Images_name + str(i) + '.png')

    for adr in file_names:
        images.append(imageio.imread(adr))

    imageio.mimsave(output_name + '.gif', images, fps=FPS)