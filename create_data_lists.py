from utils import create_data_lists
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder", help="data folder path where images and annotations are stored", type=str)
    parser.add_argument("output_folder", help="output folder path", type=str)
    parser.add_argument("-r", "--val_ratio", type=float, help="percentage of images to keep for validation")
    args = parser.parse_args()

    args.__dict__ = {k: v for k, v in args.__dict__.items() if v is not None}
    create_data_lists(**vars(args))
