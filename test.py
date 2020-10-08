from model import Model

import tensorflow as tf
import numpy as np
import argparse
import cv2
import pdb
import os

def main(args):
    ### create model
    model = Model()
    model.create_model()
    model.load_model()
    ### load image
    image = cv2.imread(args.inp_path)
    res = model.inference(image)
    print(res)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_path', default='/home/dunglt/cmnd/dung/data/process_extract_0606/extract_0606_id_number/valid/2019_03_04_19_0304_000_003_476.jpg')
    args = parser.parse_args()
    main(args)