# -*- coding: utf-8 -*-
import logging
import os
import pdb
from glob import glob
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    test_images = torch.tensor(np.load(input_filepath + "/test.npz")["images"])
    test_labels = torch.tensor(np.load(input_filepath + "/test.npz")["labels"])

    test_dataset = torch.utils.data.TensorDataset(
        test_images, test_labels.type(torch.LongTensor)
    )

    train_files = glob(input_filepath + "/train*.npz")

    train_images = np.load(train_files[0])["images"]
    train_labels = np.load(train_files[0])["labels"]
    for i in range(1, 5):
        train_images = np.concatenate([train_images, np.load(train_files[i])["images"]])
        train_labels = np.concatenate([train_labels, np.load(train_files[i])["labels"]])

    train_images = torch.Tensor(train_images)
    train_labels = torch.Tensor(train_labels)

    train_dataset = torch.utils.data.TensorDataset(
        train_images, train_labels.type(torch.LongTensor)
    )

    torch.save(train_dataset, output_filepath + "/train_dataset.pt")
    torch.save(test_dataset, output_filepath + "/test_dataset.pt")

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
