import functools
import os
import sys
import tempfile
from typing import Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import sidekick
from tqdm import tqdm


def download_and_extract_zip_file(url: str, directory: str) -> None:
    if not os.path.isdir(directory):
        sys.exit('Directory provided does not exist')

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Total size in bytes.
    total_size = int(response.headers['content-length'])
    block_size = 1024

    with tempfile.TemporaryFile() as handle:
        for chunk in tqdm(
                response.iter_content(block_size),
                total=total_size//block_size,
                unit='KB',
        ):
            if chunk:  # filter out keep-alive new chunks
                handle.write(chunk)

        with ZipFile(handle) as zip_handle:
            zip_handle.extractall(directory)


def balance_dataset(df: pd.DataFrame, cat_column: str) -> pd.DataFrame:
    # oversampling of under-represented categories
    cat_counts = df[cat_column].value_counts().to_dict()
    larger_class = max(cat_counts, key=cat_counts.get)
    counts_larger_class = cat_counts[larger_class]

    for cat in cat_counts:
        samples = counts_larger_class - cat_counts[cat]
        sampling = df[df[cat_column] == cat].sample(samples, replace=True)
        df = pd.concat([df, sampling], ignore_index=True)

    # shuffle dataset
    return df.sample(frac=1).reset_index(drop=True)


def create_ham_dataset(
        directory: str = None,
        size: Tuple[int, int] = (224, 224),
        split: float = 0.8,
        balance: bool = True,
        ) -> None:
    """
    Creates zip file with the HAM10000 dataset ready to be uploaded to 
    the Peltarion platform. The dataset contains labeled images of different 
    types of skin lesions. Read more here: https://arxiv.org/abs/1803.10417.
    
    All data for this task is provided under the terms of the Creative Commons 
    Attribution-NonCommercial (CC BY-NC) 4.0 license. You may find the terms of 
    the licesence here: https://creativecommons.org/licenses/by-nc/4.0/legalcode.
    If you are unable to accept the terms of this license, do not download or 
    use this data.
    
    Disclaimer for dataset: please notice that the disclaimer in 
    the README.md applies.

    Arguments:
    directory:  Directory where the dataset will be stored. If not provided, 
                it defaults to the current working directory.
    size:       Image size after resizing: (width, height). The original 
                image size is (600, 450).
    split:      Split fraction between training and validation. 
    balance:    Balance training dataset by oversampling under-represented
                categories
    """""

    images_dir = 'ISIC2018_Task3_Training_Input'
    metadata_dir = 'ISIC2018_Task3_Training_GroundTruth'
    metadata_file = 'ISIC2018_Task3_Training_GroundTruth.csv'

    metadata_zip_url = 'https://challenge.kitware.com/api/v1/item/5ac20eeb56357d4ff856e136/download'
    images_zip_url = 'https://challenge.kitware.com/api/v1/item/5ac20fc456357d4ff856e139/download'

    if directory is None:
        directory = os.getcwd()

    if not os.path.isdir(directory):
        sys.exit('Directory provided does not exist')

    dataset_path = os.path.join(directory, 'ham_dataset.zip')

    with tempfile.TemporaryDirectory() as tmpdir:
        download_and_extract_zip_file(metadata_zip_url, tmpdir)
        download_and_extract_zip_file(images_zip_url, tmpdir)

        # read metadata
        df = pd.read_csv(os.path.join(tmpdir, metadata_dir, metadata_file))

        # decode one-hot encoding
        categories = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        df['category'] = df[categories].idxmax(axis=1)
        df = df.drop(categories, axis=1)

        # split dataset into train and validation
        df['subset'] = ['train' if np.random.random_sample() < split else 'val'
                        for _ in range(df.shape[0])]

        if balance:
            train = df[df['subset'] == 'train']
            val = df[df['subset'] == 'val']
            train_balanced = balance_dataset(train, cat_column='category')
            df = pd.concat([train_balanced, val], ignore_index=True)

        # replace image name by image path
        df['image'] = df['image'].apply(
            lambda x: os.path.join(tmpdir, images_dir, x + '.jpg'))

        image_processor = functools.partial(
            sidekick.process_image,
            mode='resize',
            size=size,
            format='jpeg'
        )

        print('Creating dataset...')
        sidekick.create_dataset(
            dataset_path,
            df,
            path_columns=['image'],
            preprocess={'image': image_processor},
            progress=True,
        )
