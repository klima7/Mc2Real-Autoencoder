from glob import glob
import os

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Resize
from PIL import Image
from tqdm import tqdm


class DirImagesDataset(Dataset):
    
    def __init__(self, dir_path, target, size, limit=None, cache_path=None):
        self.images = self.__get_images(dir_path, limit, size, cache_path)
        self.target = np.float32(target)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        return self.images[index]-0.5, self.target
    
    def __get_images(self, dir_path, limit, size, cache_path):
        if cache_path and os.path.exists(cache_path+'.npy'):
            print('loading cached images')
            return self.__load_cached_images(cache_path, size, limit)
        else:
            print('loading images from dataset')
            images = self.__load_images(dir_path, limit, size)
            if cache_path:
                self.__save_images_to_cache(cache_path, images)
            return images
        
    def __load_cached_images(self, cache_path, size, limit):
        images = np.load(cache_path+'.npy')
        assert images.shape[2] == images.shape[3] == size
        return images
        
    def __save_images_to_cache(self, cache_path, images):
        np.save(cache_path, images)
        
    def __load_images(self, dir_path, limit, size):
        paths = glob(f'{dir_path}/**/*.png', recursive=True)[:limit]
        images = [self.__load_image(path, size) for path in tqdm(paths)]
        return np.array(images)

    def __load_image(self, path, size):
        image = Image.open(path)
        resize = Resize(size)
        image = np.array(resize(image), dtype=np.float32) / 255
        image = np.transpose(image, [2, 0, 1])
        return image
