from model import Model
from hparams import hparams
from utils import *

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
    model.load_model(args.ckpt_path)
    ### load image
    image = cv2.imread(args.inp_path)
    
    # resize image
    image = image/255.
    if image.shape != hparams.image_shape:
        h, w, d = image.shape
        max_w = hparams.image_shape[1]
        max_h = hparams.image_shape[0]
        unpad_im = cv2.resize(image, (int(max_h*w/h), max_h), interpolation = cv2.INTER_AREA)
        if unpad_im.shape[1] > max_w:
            pad_im = cv2.resize(image, (max_w, max_h), interpolation = cv2.INTER_AREA)
        else:
            pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(max_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
    else:
        pad_im = image

    # get saliency map
    os.makedirs('inference_logs', exist_ok=True)
    output = np.zeros((1, 1))
    for i in range(hparams.max_char_length):
        with tf.GradientTape() as tape:
            image_var = tf.Variable(initial_value=np.array([pad_im]), trainable=True, dtype=float)
            encoder_input = model.input_embedding_layer(image_var)
            enc_padding_mask = None
            combined_mask = create_look_ahead_mask(i+1)
            dec_padding_mask = None

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = model.transformer(encoder_input, 
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            output = tf.concat([output[:, :-1], predicted_id, output[:, -1:]], axis=-1)
            # get saliency map
            char_index = predicted_id.numpy()[0][0]
            loss = predictions[0][-1][char_index]
            grads = tape.gradient(loss, image_var)
            dgrad_abs = tf.math.abs(grads)
            dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
            arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
            grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
            cv2.imwrite('inference_logs/' + str(i) + '_saliency.jpg', grad_eval*255.)
    print(output.numpy()[0][:-1])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_path', default='test_image.jpg')
    parser.add_argument('--ckpt_path', default='training_checkpoints/train/ckpt-17')
    args = parser.parse_args()
    main(args)