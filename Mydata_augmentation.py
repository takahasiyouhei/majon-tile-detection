import numpy as np
from keras.preprocessing.image import ImageDataGenerator



def save_da_images(image, batch_size=10, save_to_dir=None):
    '''
    :param images: Numpy array of rank 4 (image_num, height, width, channels)
    :param batch_size: int
    :param save_to_dir: str.
    :return: None
    '''
    datagen = ImageDataGenerator(
        rotation_range=1,
        width_shift_range=0.10,
        height_shift_range=0.05,
        shear_range=1, zoom_range=0.1)

    gen = datagen.flow(image, batch_size=batch_size, save_to_dir=save_to_dir, save_prefix='da')

    if save_to_dir:
        for i in range(batch_size):
            gen_img = next(gen)


if __name__ == '__main__':
    import numpy as np
    from keras.preprocessing.image import load_img, img_to_array
    import os 
    for i in range(1):
        i += 1
        path = './images/temp_test(36,24)/' + str(i) + '-'
        dst_path = './images/temp_test(36,24)_da/' + str(i) #保存先のディレクトリパス
        img_list = os.listdir(path) #ファイル内の画像パスをリストで取得
        for i in img_list:
            image = load_img(os.path.join(path, i))
            x = img_to_array(image)[np.newaxis]  # (Height, Width, Channels)  -> (1, Height, Width, Channels)

            import os
            os.makedirs(dst_path, exist_ok=True)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)

            save_da_images(image=x, batch_size=5, save_to_dir=dst_path)





