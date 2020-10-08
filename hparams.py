
class Hparams:
    def __init__(self):
        ### data and save path
        # path to train record file
        self.train_record_path = '/home/common_gpu2/local_storage/dunglt/cmnd/extract_0606_id_number_v2/extract_0606_id_number_v2.valid'
        # number of samples in train record file
        self.num_train_sample = 82324
        # path to valid record file
        self.valid_record_path = '/home/common_gpu2/local_storage/dunglt/cmnd/extract_0606_id_number_v2/extract_0606_id_number_v2.valid'
        # number of samples in valid record file
        self.num_valid_sample = 3213
        # path to characters file
        self.charset_path = 'charsets/charset_size=11.txt'
        # path to save models
        self.save_path = 'training_checkpoints'
        # save only best model or not
        self.save_best = True
        self.max_to_keep = 1000

        ### input params
        # input image shape after resize and padding
        self.image_shape = (64, 500, 3)
        # index of null code in character list
        self.nul_code = 10
        # number of characters
        self.charset_size = 11
        # max length of output text
        self.max_char_length = 12

        ### embedding layer params
        # base model from tf.keras.application, or custom instance of tf.keras.Model
        # check for new models from https://www.tensorflow.org/api_docs/python/tf/keras/applications
        # check for newest model from tf-nightly version
        self.base_model_name = 'InceptionResNetV2'
        # last convolution layer from base model which extract features from
        self.end_point = 'mixed_6a'
        # endcode cordinate feature to conv_feature
        self.use_encode_cordinate = False

        ### transformer params follow https://arxiv.org/pdf/1706.03762.pdf
        self.use_input_position_encode = True
        self.use_output_position_encode = True
        self.num_heads = 8
        self.num_layers = 6
        self.model_size = 512
        self.dropout_rate = 0.1

        ### training params
        self.batch_size = 32
        self.max_epochs = 100
        self.learning_rate = 0.0001 # schedule

hparams = Hparams()
