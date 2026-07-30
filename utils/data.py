import tensorflow as tf
import os
from io import BytesIO
from zipfile import ZipFile
from sklearn.utils import shuffle

data_path = "/path/to/data"
data_file = "/data/file"
file_path = data_path + data_file

class DataLoader():
    def __init__(self, csv_file=file_path, DEBUG=False):
        self.shape_rgb = (480, 640, 3)
        self.shape_depth = (240, 320, 1)
        self.filenames = []
        self.labels = []
        self.filenames1 = []
        self.labels1 = []
        self.train_filenames1 = []
        self.train_labels1 = []
        self.val_filenames1 = []
        self.val_labels1 = []
        self.read_nyu_data(csv_file, DEBUG=DEBUG)

    def nyu_resize(self, img, resolution=480, padding=6):
        from skimage.transform import resize
        return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True)

    def read_nyu_data(self, csv_file, DEBUG=False):
        csv = open(csv_file, 'r').read()
        nyu2_train = list((row.split(',') for row in (csv).split('\n') if len(row) > 0))

        print("nyu2_train : ", nyu2_train)

        # Dataset shuffling happens here
        nyu2_train = shuffle(nyu2_train, random_state=0)

        # Test on a smaller dataset
        if DEBUG:
            nyu2_train = nyu2_train[:10]

        # A vector of RGB filenames.
        self.filenames = [i[0] for i in nyu2_train]

        for file in self.filenames:
            path = os.path.join(data_path, file)
            self.filenames1.append(path)

        # A vector of depth filenames.
        self.labels = [i[1] for i in nyu2_train]

        for label in self.labels:
            path = os.path.join(data_path, label)
            self.labels1.append(path)

        # Split data into training and validation sets (80-20 split)
        split_index = int(0.8 * len(self.filenames1))
        self.train_filenames1 = self.filenames1[:split_index]
        self.train_labels1 = self.labels1[:split_index]
        self.val_filenames1 = self.filenames1[split_index:]
        self.val_labels1 = self.labels1[split_index:]

        # Length of dataset
        self.length = len(self.train_filenames1)

    def _parse_function(self, filename, label):
        # Read images from disk
        image_decoded = tf.image.decode_jpeg(tf.io.read_file(filename))
        depth_resized = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(label)),
                                        [self.shape_depth[0], self.shape_depth[1]])

        # Format
        rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
        depth = tf.image.convert_image_dtype(depth_resized / 255.0, dtype=tf.float32)

        # Normalize the depth values (in cm)
        depth = 1000 / tf.clip_by_value(depth * 1000, 10, 1000)

        return rgb, depth

    def get_batched_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((self.train_filenames1, self.train_labels1))
        dataset = dataset.shuffle(buffer_size=len(self.train_filenames1), reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(map_func=self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size)
        return dataset

    def get_validation_dataset(self, batch_size):
        validation_dataset = tf.data.Dataset.from_tensor_slices((self.val_filenames1, self.val_labels1))
        validation_dataset = validation_dataset.map(map_func=self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.batch(batch_size=batch_size)
        return validation_dataset
