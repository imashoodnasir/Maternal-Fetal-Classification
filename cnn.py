import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D

class CNNBlock(tf.keras.Model):
    def __init__(self):
        super(CNNBlock, self).__init__()

        # Define convolutional layers with increasing filter sizes
        self.conv1 = Conv2D(32, (3, 3), padding='same', activation=None)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.pool1 = MaxPooling2D((2, 2))

        self.conv2 = Conv2D(32, (3, 3), padding='same', activation=None)
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.pool2 = MaxPooling2D((2, 2))

        self.conv3 = Conv2D(64, (3, 3), padding='same', activation=None)
        self.bn3 = BatchNormalization()
        self.relu3 = ReLU()
        self.pool3 = MaxPooling2D((2, 2))

        self.conv4 = Conv2D(128, (3, 3), padding='same', activation=None)
        self.bn4 = BatchNormalization()
        self.relu4 = ReLU()
        self.pool4 = MaxPooling2D((2, 2))

    def call(self, inputs):
        # Pass through each convolutional block
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        return x
