import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data_loader.utils import Input2dSpec, LABEL_FRACS


class Jinchi(Dataset):
    '''A dataset class for the ophthalmology dataset from the publication 
    "Applying artificial intelligence to disease staging: Deep learning for improved staging of diabetic retinopathy, 
    grading diabetic retinopathy on the Davis Scale."
    (https://figshare.com/articles/figure/Davis_Grading_of_One_and_Concatenated_Figures/4879853/1)
    Note: 
        1) A modified Davis scale is used, with classes "No Disease", "Simple DR", "Pre-Proliferative DR", and "Proliferative DR."
        The standard Davis scale runs from 0-4, inclusive, where 0 corresponds to "No Disease", 1 and 2 corresponds to "Simple DR", 
        3 corresponds to "Pre-Proliferative DR", and 4 corresponds to "Proliferative DR."
        2) You must manually download the data to use this dataset. Move the downloaded "dmr" folder to this directory.
    '''

    # Dataset information.
    NUM_CLASSES = 4  # 4 classes in modified Davis scale.
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 3
    RANDOM_SEED = 0
    TRAIN_SPLIT_FRAC = 0.8

    def __init__(self, base_root: str, train: bool = True, transform = None) -> None:
        super().__init__()
        self.root = os.path.join(base_root, 'fundus', 'dmr')
        self.split = 'train' if train else 'val'
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        # self.transforms = transforms.Compose(
        #     [
        #         transforms.Resize(self.INPUT_SIZE[0]),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.4934, 0.2844, 0.1583], [0.2896, 0.1822, 0.1158]),
        #     ]
        # )
        self.transform = transform
        self.build_index()

    def convert_labels(self, row):
        letter_label = row['Davis_grading_of_one_figure']
        if letter_label == "ndr":
            row['Davis_grading_of_one_figure'] = 0
        elif letter_label == "sdr":
            row['Davis_grading_of_one_figure'] = 1
        elif letter_label == "ppdr":
            row['Davis_grading_of_one_figure'] = 2
        elif letter_label == "pdr":
            row['Davis_grading_of_one_figure'] = 3
        else:
            raise ValueError('Label not recognized.')
        return row

    def build_index(self):
        print('Building index...')
        image_info = os.path.join(self.root, 'list.csv')

        # load Image and Davis_grading_of_one_figure columns
        df = pd.read_csv(image_info, header=0, usecols=[0, 2])
        df = df.apply(lambda row: self.convert_labels(row), axis=1)  # convert letter-labels to numbers

        # manually create splits
        unique_counts = df['Davis_grading_of_one_figure'].value_counts()
        train_df = pd.DataFrame(columns=df.columns)

        for label, count in unique_counts.items():
            num_sample = int(Jinchi.TRAIN_SPLIT_FRAC * count)
            train_rows = df.loc[df['Davis_grading_of_one_figure'] == label].sample(num_sample, random_state=Jinchi.RANDOM_SEED)
            if self.split == 'train':
                train_df = train_df.append(train_rows)
            else:
                df = df.drop(train_rows.index)

        df = train_df if self.split == 'train' else df

        self.fnames = df['Image'].to_numpy(dtype=np.str)
        self.labels = df['Davis_grading_of_one_figure'].to_numpy(dtype=np.str)

        print('Done \n\n')

    def __getitem__(self, index):
        img_path, label = os.path.join(self.root, self.fnames[index]), int(self.labels[index])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label, index

    def __len__(self):
        return self.fnames.shape[0]

    @staticmethod
    def num_classes():
        return Jinchi.NUM_CLASSES

    @staticmethod
    def spec():
        '''Returns a dict containing dataset spec.'''
        return [
            Input2dSpec(input_size=Jinchi.INPUT_SIZE, patch_size=Jinchi.PATCH_SIZE, in_channels=Jinchi.IN_CHANNELS),
        ]
