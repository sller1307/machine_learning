import json
from PIL import Image
from os import listdir
import pickle
import time
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import collections
from copy import deepcopy

ANNOTATION_DIR = "../data/annotations/captions_train2017.json"
IMAGE_DIR = "../data/train2017/"
SAVE_PATH = "../model/"
CACHE_DIR = "../cache/"


def load_json(path):
    with open(path, "r") as f:
        output_dict = json.load(f)

    return output_dict


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)
        )

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class ImageCaption:
    def __init__(self):

        self.features_extractor = self._build_features_extractor_model()

    @staticmethod
    def _build_features_extractor_model():
        image_model = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet"
        )
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output

        return tf.keras.Model(new_input, hidden_layer)

    def _build_image_caption_model(self):

        embedding_dim = 256
        units = 512
        vocab_size = 5001

        encoder = CNN_Encoder(embedding_dim)
        decoder = RNN_Decoder(embedding_dim, units, vocab_size)

        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = tf.keras.optimizers.Adam()

    def fit_on_data(self, data_manager):
        self.max_length = data_manager.max_length
        self.tokenizer = deepcopy(data_manager.tokenizer)

    @tf.function
    def train_step(self, img_tensor, target):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)

        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)

                loss += loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def train(self, dataset):
        checkpoint_path = "../checkpoints/train"
        ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                   decoder=self.decoder,
                                   optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        start_epoch = 0
        if ckpt_manager.latest_checkpoint:
            start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            # restoring the latest checkpoint in checkpoint_path
            ckpt.restore(ckpt_manager.latest_checkpoint)

        EPOCHS = 20

        for epoch in range(start_epoch, EPOCHS):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')

            if epoch % 5 == 0:
                ckpt_manager.save()

            print(f'Epoch {epoch + 1} Loss {total_loss / len(dataset):.6f}')
            print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    def generate_caption(self, image):

        hidden = self.decoder.reset_state(batch_size=1)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image, (299, 299))
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        img_tensor_val = self.features_extractor(image)

        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                     -1,
                                                     img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input,
                                                             features,
                                                             hidden)

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)

        return result


class DataManager:
    def __init__(
        self,
        num_of_data=None,
        image_dir=IMAGE_DIR,
        annotation_dir=ANNOTATION_DIR,
        cache_path=CACHE_DIR,
    ):
        self.image_dir = image_dir
        self.annotation_path = annotation_dir
        self.cache_path = cache_path

        self._build_image_ID_to_caption(self.annotation_path, num_of_data)

    def _build_image_ID_to_caption(self, annotation_dir, num_of_data=None):
        annotations = load_json(annotation_dir)["annotations"]
        self.imageID_to_caption = collections.defaultdict(list)

        for annotation in annotations:
            self.imageID_to_caption["%012d" % annotation["image_id"]].append(
                self.add_start_and_end_to_caption(annotation["caption"])
            )

        if num_of_data is not None:
            self.imageID_to_caption = self.head(num_of_data)
        self.imageIDs = list(self.imageID_to_caption.keys())

    def head(self, n):
        filtered_imageIDs = list(self.imageID_to_caption.keys())[:n]

        return {
            imageID: caption
            for imageID, caption in self.imageID_to_caption.items()
            if imageID in filtered_imageIDs
        }

    def imageID_to_imagepath(self, imageID):
        return self.image_dir + imageID

    def get_image(self, imageID):
        return Image.open(self.imageID_to_imagepath(imageID))

    def process_and_cache_images(self, feature_extractor):
        def _load_image(imageID):
            image_path = (
                tf.convert_to_tensor(self.image_dir)
                + imageID
                + tf.convert_to_tensor(".jpg")
            )
            img = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, (299, 299))
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            return img, imageID

        image_data = tf.data.Dataset.from_tensor_slices(self.imageIDs)
        image_data = image_data.map(
            _load_image, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(16)

        for image, imageIDs in tqdm(image_data):
            batch_features = feature_extractor(image)
            batch_features = tf.reshape(
                batch_features, (batch_features.shape[0], -1, batch_features.shape[3])
            )

            for batch_feature, imageID in zip(batch_features, imageIDs):
                path_of_feature = self.cache_path + imageID.numpy().decode("utf-8")
                np.save(path_of_feature, batch_feature.numpy())

    def process_captions(self):
        imageIDs = []
        train_captions = []
        for imageID, captions in self.imageID_to_caption.items():
            imageIDs.extend([imageID] * len(captions))
            train_captions.extend(captions)

        top_k = 5000
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~'
        )
        tokenizer.fit_on_texts(train_captions)
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"
        train_seqs = tokenizer.texts_to_sequences(train_captions)
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
            train_seqs, padding="post"
        )

        self.tokenizer = tokenizer
        self.max_length = self.calc_max_length(train_seqs)
        return imageIDs, cap_vector

    def build_dataset(self):
        BATCH_SIZE = 64
        BUFFER_SIZE = 1000

        def map_func(imageID, caption):
            img_tensor = np.load(self.cache_path + imageID.decode("utf-8") + ".npy")
            return img_tensor, caption

        imageIDs, cap_vector = self.process_captions()

        dataset = tf.data.Dataset.from_tensor_slices((imageIDs, cap_vector))
        dataset = dataset.map(
            lambda item1, item2: tf.numpy_function(
                map_func, [item1, item2], [tf.float32, tf.int32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.model_input = dataset

        return self.model_input

    def calc_max_length(self, tensor):
        return max(len(t) for t in tensor)

    def add_start_and_end_to_caption(self, caption):
        return "<start> " + caption + " <end>"
