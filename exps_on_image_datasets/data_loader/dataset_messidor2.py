import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader.utils import Input2dSpec, LABEL_FRACS


class Messidor2(Dataset):
    '''A dataset class for the Messidor 2 dataset, ophthalmology dataset, grading diabetic retinopathy on the 0-4 Davis Scale.
    (https://www.adcis.net/en/third-party/messidor2/)
    (https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)
    Note that you must register and manually download the data to use this dataset. Download the main folder from the adcis.net 
    link, and extract all the files to a "messidor2" folder in this directory. It should contain messidor-2.csv and an "IMAGES" directory.
    Then add the messidor_data.csv and messidor_readme.txt files from the kaggle link to the "messidor2" directory as well.
    '''

    # Dataset information.
    NUM_CLASSES = 5
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3
    TRAIN_SPLIT_FRAC = 0.8
    RANDOM_SEED = 0

    def __init__(self, base_root: str, train: bool = True, finetune_size: str = 'full', transform = None) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'fundus', 'messidor2')
        self.split = 'train' if train else 'test'
        self.finetune_size = 0 if finetune_size is None else LABEL_FRACS[finetune_size]
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        # self.transforms = [
        #         transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),  #resizes (H,W) to (149, 224)
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.2859, 0.1341, 0.0471], [0.3263, 0.1568, 0.0613]),
        #         transforms.Pad((0, 37, 0, 38))
        #     ]
        
        self.transform = transform 
        self.build_index()

    def edit_ext(self, row):
        if row['image_id'][-3:] == "jpg":
            row['image_id'] = row['image_id'][:-3] + 'JPG'
        return row

    def build_index(self):
        print('Building index...')
        image_info = os.path.join(self.root, 'messidor_data.csv')

        # load image_id and adjudicated_dr_grade columns
        df = pd.read_csv(image_info, header=0, usecols=[0, 1])
        df = df.dropna()  # some rows don't have diagnosis info
        df = df.apply(lambda row: self.edit_ext(row), axis=1)  # capitalize 'jpg' extensions to match file names

        # manually create splits
        # first, get counts for every label
        unique_counts = df['adjudicated_dr_grade'].value_counts()
        train_df = pd.DataFrame(columns=df.columns)

        for label, count in unique_counts.items():
            # if finetuning, get 'finetune_size' labels for each class
            # if insufficient examples, use all examples from that class
            # otherwise, use 80% of examples for training, other 20% for validation
            if self.split == 'train' and self.finetune_size > 0:
                num_sample = min(self.finetune_size, count)
            else:
                num_sample = int(Messidor2.TRAIN_SPLIT_FRAC * count)
            train_rows = df.loc[df['adjudicated_dr_grade'] == label].sample(num_sample, random_state=Messidor2.RANDOM_SEED)
            # if training, add sampled examples to train_df. otherwise drop them to make validation set
            if self.split == 'train':
                train_df = train_df.append(train_rows)
            else:
                df = df.drop(train_rows.index)

        # if training, we create new dataframe from selected examples. otherwise, we dropped those examples.
        if self.split == 'train':
            df = train_df

        self.fnames = df['image_id'].to_numpy(dtype=np.str)
        self.labels = df['adjudicated_dr_grade'].to_numpy(dtype=np.str)

        print('Done \n\n')

    def __getitem__(self, index):
        img_path, label = os.path.join(self.root, 'IMAGES', self.fnames[index]), int(self.labels[index][0])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, index

    def __len__(self):
        return self.fnames.shape[0]

    @staticmethod
    def num_classes():
        return Messidor2.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=Messidor2.INPUT_SIZE, patch_size=Messidor2.PATCH_SIZE, in_channels=Messidor2.IN_CHANNELS),
        ]
