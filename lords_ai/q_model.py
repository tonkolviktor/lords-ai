# https://keras.io/examples/rl/deep_q_network_breakout/

from tensorflow.keras import layers, models


def create_q_model(num_actions: int):
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(14,))

    # Convolutions on the frames on the screen
    #layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    #layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    #layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    #layer4 = layers.Flatten()(layer3)
    hidden = layers.BatchNormalization()(inputs)
    hidden = layers.Dense(28, activation="relu")(hidden)
    hidden = layers.Dense(56, activation="relu")(hidden)
    hidden = layers.Dense(56, activation="relu")(hidden)
    hidden = layers.Dense(28, activation="relu")(hidden)
    action = layers.Dense(num_actions, activation="linear")(hidden)

    return models.Model(inputs=inputs, outputs=action)