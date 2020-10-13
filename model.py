from custom_layers.input_embeddings import InputEmbedding
from custom_layers.learning_rate import CustomSchedule
from custom_layers.transformer import Transformer
from datetime import datetime
from dataset import Dataset
from hparams import hparams
from tqdm import tqdm
from utils import *

import tensorflow as tf
import numpy as np
import logging
import cv2
import pdb
import os
logging.basicConfig(level=logging.DEBUG)

class Model(object):
    def __init__(self):
        self.char_mapping = {}
        with open(hparams.charset_path, 'r') as f:
            for row in f:
                index, label = row[:-1].split('\t')
                self.char_mapping[int(index)] = label

    def loss_function(self, real, pred):
        loss_ = self.loss_object(real, pred)
        return loss_

    def create_model(self):
        ### create model
        self.best_val_acc = 0.0
        # dataset
        self.train_dataset = Dataset(hparams, hparams.train_record_path)
        self.valid_dataset = Dataset(hparams, hparams.valid_record_path)
        self.train_dataset.load_tfrecord()
        self.valid_dataset.load_tfrecord()
        # create input embedding layer
        self.input_embedding_layer = InputEmbedding(hparams)
        # create attention + RNN layer
        self.look_ahead_mask = create_look_ahead_mask(hparams.max_char_length)
        self.transformer = Transformer(hparams, self.input_embedding_layer.conv_out_shape)
        
        ### define training ops and params
        if hparams.learning_rate == 'schedule':
            self.learning_rate = CustomSchedule(hparams.model_size)
        else:
            self.learning_rate = hparams.learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.last_epoch = 0
        self.train_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/train')
        self.valid_summary_writer = tf.summary.create_file_writer(hparams.save_path + '/logs/valid')
        self.checkpoint_dir = os.path.join(hparams.save_path, 'train')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              input_embedding=self.input_embedding_layer,
                                              transformer=self.transformer)

    def load_model(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest != None:
            logging.info('load model from {}'.format(latest))
            self.last_epoch = int(latest.split('-')[-1])
            self.checkpoint.restore(latest)

    def train_step(self, batch_input, batch_target):
        loss = 0
        current_batch_size = batch_input.shape[0]
        with tf.GradientTape() as tape:
            input_embeddings = self.input_embedding_layer(batch_input)
            predictions, attention_weights = self.transformer(input_embeddings, batch_target, True, None, self.look_ahead_mask, None)
            loss = self.loss_function(batch_target, predictions)
        variables = self.input_embedding_layer.trainable_variables + self.transformer.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def evaluate(self, batch_input, batch_target):
        encoder_input = self.input_embedding_layer(batch_input)
        output = np.zeros((batch_input.shape[0], 1))
        for i in range(hparams.max_char_length):
            enc_padding_mask = None
            combined_mask = create_look_ahead_mask(i+1)
            dec_padding_mask = None

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, 
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            output = tf.concat([output[:, :-1], predicted_id, output[:, -1:]], axis=-1)
        batch_true_char = np.sum(np.equal(output[:, :-1], batch_target))
        batch_true_str  = np.sum(np.prod(np.equal(output[:, :-1], batch_target), axis=1))
        return batch_true_char, batch_true_str


    def inference(self, image):
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
        # run inference
        encoder_input = self.input_embedding_layer(np.array([pad_im]))
        output = np.zeros((1, 1))
        for i in range(hparams.max_char_length):
            enc_padding_mask = None
            combined_mask = create_look_ahead_mask(i+1)
            dec_padding_mask = None

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input, 
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            output = tf.concat([output[:, :-1], predicted_id, output[:, -1:]], axis=-1)
        return output


    def train(self):
        self.load_model()
        for epoch in range(self.last_epoch, hparams.max_epochs):
            total_loss = 0
            # train each batch in dataset
            for batch, (batch_input, batch_target) in enumerate(self.train_dataset.dataset):
                start = datetime.now()
                batch_loss = self.train_step(batch_input, batch_target)
                total_loss += batch_loss
                if batch % 1 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Time {}'.format(epoch + 1, batch, batch_loss, datetime.now()-start))

            # evaluate on train set
            # logging.info('evaluate on train set')
            #cnt_true_char = 0
            #cnt_true_str = 0
            #sum_char = 0
            #sum_str = 0
            #for batch, (batch_input, batch_target) in tqdm(enumerate(self.train_dataset.dataset)):
            #    batch_true_char, batch_true_str = self.evaluate(batch_input, batch_target)
            #    cnt_true_char += batch_true_char
            #    cnt_true_str  += batch_true_str
            #    sum_char += batch_input.shape[0] * hparams.max_char_length
            #    sum_str  += batch_input.shape[0]
            #train_char_acc = cnt_true_char/sum_char
            #train_str_acc  = cnt_true_str/sum_str

            # evaluate on valid set
            logging.info('evaluate on valid set')
            cnt_true_char = 0
            cnt_true_str = 0
            sum_char = 0
            sum_str = 0
            for batch, (batch_input, batch_target) in tqdm(enumerate(self.valid_dataset.dataset)):
                batch_true_char, batch_true_str = self.evaluate(batch_input, batch_target)
                cnt_true_char += batch_true_char
                cnt_true_str  += batch_true_str
                sum_char += batch_input.shape[0] * hparams.max_char_length
                sum_str  += batch_input.shape[0]
            valid_char_acc = cnt_true_char/sum_char
            valid_str_acc  = cnt_true_str/sum_str
            # save checkpoint
            
            if hparams.save_best:
                if self.best_val_acc < valid_str_acc:
                    self.checkpoint.save(file_prefix = self.checkpoint_prefix)
            else:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            # write log
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', total_loss, step=epoch)
            #    tf.summary.scalar('character accuracy', train_char_acc, step=epoch)
            #    tf.summary.scalar('sequence accuracy', train_str_acc, step=epoch)

            with self.valid_summary_writer.as_default():
                tf.summary.scalar('character accuracy', valid_char_acc, step=epoch)
                tf.summary.scalar('sequence accuracy', valid_str_acc, step=epoch)

            # log traing result of each epoch
            logging.info('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / batch))
            #logging.info('Accuracy on train set:')
            #logging.info('character accuracy: {:.6f}'.format(train_char_acc))
            #logging.info('sequence accuracy : {:.6f}'.format(train_str_acc))
            logging.info('Accuracy on valid set:')
            logging.info('character accuracy: {:.6f}'.format(valid_char_acc))
            logging.info('sequence accuracy : {:.6f}'.format(valid_str_acc))
            # logging.info('Time taken for 1 epoch {} sec\n'.format(datetime.now() - start))
