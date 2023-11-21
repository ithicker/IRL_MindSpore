
from PIL import Image
from mindspore.dataset import transforms 
from mindspore.dataset import vision
import numpy as np
import os




class AgeDB():
    def __init__(self, df, data_dir, img_size, number, interval, split='train'):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size

        self.split = split
        self.label = np.asarray(df['age']).astype('float32')
        self.number = number
        self.interval = interval

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        #transform = self.get_transform()
        #img = transform(img)
        if self.split == 'train':
            return img, self.interval[index]#, self.label[index], self.number[index]
        else:
            return img, self.label[index]

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                vision.Resize((self.img_size, self.img_size)),
                vision.RandomCrop(self.img_size, padding=16),
                vision.RandomHorizontalFlip(),
                vision.ToTensor(),
                vision.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                vision.Resize((self.img_size, self.img_size)),
                vision.ToTensor(),
                vision.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

