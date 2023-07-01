from glob import glob

import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image


class ImagesFromDirectoryDataset(Dataset):
    
    def __init__(self, dir_path):
        self.paths = list(glob(f'{dir_path}/**/*.png', recursive=True))
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        image = np.array(image, dtype=np.float32) / 255
        return image
