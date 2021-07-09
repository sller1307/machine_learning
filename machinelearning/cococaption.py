import json
from PIL import Image
from os import listdir
import pickle
import tensorflow as tf
from tqdm import tqdm
import numpy as np

ANNOTATION_PATH = "../data/annotations/captions_train2017.json"
IMAGE_PATH = "../data/train2017/"
SAVE_PATH = "../model/"
CACHE_DIR = "../cache/"


def load_json(path):
    with open(path, "r") as f:
        output_dict = json.load(f)

    return output_dict


class ImageCaption:
    def __init__(
        self,
        image_path=IMAGE_PATH,
        annotation_path=ANNOTATION_PATH,
        cache_path=CACHE_DIR,
        save_path=SAVE_PATH,
    ):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.cache_path = cache_path
        self.save_path = save_path

        self.features_extractor = self._build_features_extractor_model()

        self.DataManager = DataManager(image_path, annotation_path, cache_path)

    def process_data(self, num_of_data=None):
        self.DataManager.process_annotations(
            self.features_extractor, num_of_annotations=num_of_data
        )
        self._build_image_caption_model()

    @staticmethod
    def _build_features_extractor_model():
        model = tf.keras.applications.vgg19.VGG19(include_top=True, weights="imagenet")
        return tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

    def _build_image_caption_model(self):

        maxlen = self.DataManager.maxlen
        vocab_size = self.DataManager.vocab_size

        input1 = tf.keras.Input(4096)
        fe1 = tf.keras.layers.Dropout(0.5)(input1)
        fe2 = tf.keras.layers.Dense(256, activation="relu")(fe1)

        input2 = tf.keras.Input(maxlen)
        se1 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(input2)
        se2 = tf.keras.layers.Dropout(0.5)(se1)
        se3 = tf.keras.layers.LSTM(256)(se2)

        decoder1 = tf.keras.layers.add([fe2, se3])
        decoder2 = tf.keras.layers.Dense(256, activation="relu")(decoder1)
        output = tf.keras.layers.Dense(vocab_size, activation="softmax")(decoder2)

        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        model.compile(loss="categorical_crossentropy", optimizer="adam")

        self.model = model

    def train(self, batch_size=64, epochs=20, steps_per_epochs=10000):
        data_generator = self.DataManager.data_generator(batch_size)

        for i in range(epochs):
            self.model.fit_generator(
                data_generator, epochs=1, steps_per_epoch=steps_per_epochs, verbose=1
            )

    def save_weights(self):
        self.model.save_weights(self.save_path)

    def generate_caption(self, image):
        in_text = "<start>"

        image = self.DataManager.process_image(image, self.features_extractor)
        for _ in range(self.DataManager.maxlen):
            # integer encode input sequence
            sequence = self.DataManager.tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = tf.keras.preprocessing.sequence.pad_sequences(
                [sequence], maxlen=self.DataManager.maxlen
            )
            # predict next word

            yhat = self.model.predict([image, sequence], verbose=0)
            # convert probability to integer
            yhat = np.argmax(yhat)
            # map integer to word
            word = self.DataManager.tokenizer.index_word[yhat]
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += " " + word
            # stop if we predict the end of the sequence
            if word == "<end>":
                break
        return in_text


class DataManager:
    def __init__(
        self,
        image_path=IMAGE_PATH,
        annotation_path=ANNOTATION_PATH,
        cache_path=CACHE_DIR,
    ):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.cache_path = cache_path
        self.raw_annotations = load_json(annotation_path)["annotations"]
        self.ImageIDs = self.get_ImageIDs(self.raw_annotations)

    def get_ImageIDs(self, annotations):
        return set([annotation["image_id"] for annotation in annotations])

    def process_annotations(self, features_extractor_model, num_of_annotations=None):
        imageIDs = []
        captions = []

        imageID_to_image_cache_path = {}

        # TODO: multiprocessing
        num_of_annotations = (
            len(self.raw_annotations)
            if num_of_annotations is None
            else num_of_annotations
        )

        for i in tqdm(range(num_of_annotations)):
            annotation = self.raw_annotations[i]
            imageID = annotation["image_id"]

            image = self.get_image_from_imageID(imageID)

            if imageID not in imageID_to_image_cache_path:
                if not self._is_image_cached(imageID):
                    try:
                        image_feature = self.process_image(
                            image, features_extractor_model
                        )[0]
                    except:
                        continue
                    cache_image_path = self.cache_image_feature(imageID, image_feature)
                else:
                    cache_image_path = self.cache_path + f"{imageID}.pickle"

                imageID_to_image_cache_path[imageID] = cache_image_path

            caption = self.add_start_and_end_to_caption(annotation["caption"])

            imageIDs.append(imageID)
            captions.append(caption)

        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~'
        )
        tokenizer.fit_on_texts(captions)
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"
        seqs = tokenizer.texts_to_sequences(captions)

        self.maxlen = self.find_max_length_from_sequences(seqs)
        self.vocab_size = len(tokenizer.word_index) + 1
        self.tokenizer = tokenizer

        self.processed_annotations = list(zip(imageIDs, seqs))

        return self.processed_annotations

    def find_max_length_from_sequences(self, sequences):
        max_length = -1
        for sequence in sequences:
            if len(sequence) > max_length:
                max_length = len(sequence)
        return max_length

    def cache_image_feature(self, imageID, image_feature):
        cache_image_path = self.cache_path + f"{imageID}.pickle"
        with open(cache_image_path, "wb") as handle:
            pickle.dump(image_feature, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return cache_image_path

    def get_image_from_imageID(self, imageID):
        return Image.open(self.image_path + "%012d.jpg" % imageID)

    def get_image_feature_cache_from_imageID(self, imageID):

        if self._is_image_cached(imageID):
            with open(self.cache_path + f"/{imageID}.pickle", "rb") as handle:
                image_feature = pickle.load(handle)
            return image_feature
        raise ValueError(f"{imageID} is not cached")

    def _is_image_cached(self, imageID):
        return f"{imageID}.pickle" in listdir(self.cache_path)

    def process_image(self, image, model):
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image, (224, 224))
        image = tf.expand_dims(image, axis=0)
        image = tf.keras.applications.vgg19.preprocess_input(image, data_format=None)
        return model.predict(image)

    def add_start_and_end_to_caption(self, caption):
        return "<start> " + caption + " <end>"

    def data_generator(self, batch_size):
        X1 = []
        X2 = []
        y = []
        while 1:
            for imageID, seq in self.processed_annotations:
                image = self.get_image_feature_cache_from_imageID(imageID)

                for j in range(1, len(seq)):
                    X1.append(image)
                    X2.append(seq[:j])
                    y.append(seq[j])
                    if len(X1) >= batch_size:
                        X1 = np.array(X1)
                        X2 = tf.keras.preprocessing.sequence.pad_sequences(
                            X2, maxlen=self.maxlen, padding="post"
                        )
                        y = tf.keras.utils.to_categorical(
                            y, num_classes=self.vocab_size
                        )
                        yield [X1, X2], y
                        X1 = []
                        X2 = []
                        y = []
