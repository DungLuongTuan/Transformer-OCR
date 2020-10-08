from model import Model

import tensorflow as tf
import numpy as np
import pdb
import os

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def main():
    ### create model
    model = Model()
    model.create_model()
    ### train new model
    model.train()

if __name__ == '__main__':
    main()