from model.funs import make_gif

if __name__ == '__main__':
    
    # file_path = '/photo_cGAN/photo_mnist/Images_Epochs_'\
    file_path = '/photo_cGAN/photo_cifar/Images_Epochs_'
    make_gif(file_path, 'cGAN_50epochs_cifar', top_range=50)