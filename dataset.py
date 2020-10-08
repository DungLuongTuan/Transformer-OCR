import tensorflow as tf
import math 

class Dataset(object):
    def __init__(self, hparams, record_path):
        self.hparams = hparams
        self.record_path = record_path
        zero = tf.zeros([1], dtype=tf.int64)
        self.keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.io.FixedLenFeature([], tf.string, default_value='png'),
            'image/width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/orig_width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/class':tf.io.FixedLenFeature([hparams.max_char_length], tf.int64),
            'image/unpadded_class':tf.io.VarLenFeature(tf.int64),
            'image/text': tf.io.FixedLenFeature([1], tf.string, default_value=''),
        }

    def parse_tfrecord(self, example):
        res = tf.io.parse_single_example(example, self.keys_to_features)
        image = tf.cast(tf.io.decode_jpeg(res['image/encoded'], 3), tf.float32)/255.0
        label = tf.cast(res['image/class'], tf.float32)
        return image, label

    def load_tfrecord(self, repeat=None):
        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.map(self.parse_tfrecord)
        # dataset = dataset.shuffle(5000)
        self.dataset = dataset.batch(self.hparams.batch_size)
        #self.dataset = dataset.cache()
        self.iterator = iter(dataset)

    def next_batch(self):
        return self.iterator.get_next()
