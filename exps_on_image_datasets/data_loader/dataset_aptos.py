import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader.utils import Input2dSpec, LABEL_FRACS



class Aptos(Dataset):
    '''A dataset class for the APTOS 2019 Blindness Detection dataset, grading diabetic retinopathy on the Davis Scale
    from retina images taken using fundus photography.
    (https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)
    Note that you must register for the and use the kaggle command noted in the above link to download the dataset to this directory.
    Rename the downloaded folder "aptos", and it should contain sample_submission.csv, test.csv, train.csv, and folders for 
    test_images and train_images.
    '''

    # Dataset information.
    NUM_CLASSES = 5  # Classify on Davis Scale from 0 to 4, inconclusive.
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3
    RANDOM_SEED = 0
    TRAIN_SPLIT_FRAC = 0.8

    def __init__(self, base_root: str, train: bool = True, finetune_size: str = 'full', transform = None) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'fundus', 'aptos')
        self.split = 'train' if train else 'val'
        self.finetune_size = 0 if finetune_size is None else LABEL_FRACS[finetune_size]
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        # self.transforms = transforms.Compose(
        #     [
        #         transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.4210, 0.2238, 0.0725], [0.2757, 0.1494, 0.0802]),
        #     ]
        # )
        # self.transforms_sq = transforms.Compose(
        #     [
        #         transforms.Resize(self.INPUT_SIZE[0]),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.4210, 0.2238, 0.0725], [0.2757, 0.1494, 0.0802]),
        #     ]
        # )
        self.transform = transform
        self.build_index()

    def build_index(self):
        print('Building index...')
        image_info = os.path.join(self.root, 'train.csv')

        # load id_code and diagnosis columns
        df = pd.read_csv(image_info, header=0, usecols=[0, 1])

        # manually create splits
        unique_counts = df['diagnosis'].value_counts()
        train_df = pd.DataFrame(columns=df.columns)

        for label, count in unique_counts.items():
            # if finetuning, get 'finetune_size' labels for each class
            # if insufficient examples, use all examples from that class
            # otherwise, use 80% of examples for training, other 20% for validation
            if self.split == 'train' and self.finetune_size > 0:
                num_sample = min(self.finetune_size, count)
            else:
                num_sample = int(Aptos.TRAIN_SPLIT_FRAC * count)
            train_rows = df.loc[df['diagnosis'] == label].sample(num_sample, random_state=Aptos.RANDOM_SEED)
            if self.split == 'train':
                train_df = train_df.append(train_rows)
            else:
                df = df.drop(train_rows.index)

        df = train_df if self.split == 'train' else df

        self.fnames = df['id_code'].to_numpy(dtype=np.str)
        self.labels = df['diagnosis'].to_numpy(dtype=np.str)

        print('Done \n\n')
    
    def resize_img(self, img):
        dim_gap = img.shape[2] - img.shape[1]
        pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
        img = transforms.Pad((0, pad1, 0, pad2))(img)
        img = transforms.Resize((224, 224))(img)
        return img

    def __getitem__(self, index):
        img_path, label = os.path.join(self.root, 'train_images', self.fnames[index] + '.png'), int(self.labels[index])
        img = Image.open(img_path).convert('RGB')
        # if img.size[0] == img.size[1]:
        #     img = self.transforms_sq(img)
        # else:
        #     img = self.transforms(img)
        #     # Add image-dependent padding
        #     dim_gap = img.shape[2] - img.shape[1]
        #     pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
        img = self.transform(img)
        if type(img) == tuple:
            return (self.resize_img(img[0]), self.resize_img(img[1])), label, index
        else:
            return self.resize_img(img), label, index

    def __len__(self):
        return self.fnames.shape[0]

    @staticmethod
    def num_classes():
        return Aptos.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=Aptos.INPUT_SIZE, patch_size=Aptos.PATCH_SIZE, in_channels=Aptos.IN_CHANNELS),
        ]
