import tensorflow as tf


def load_data_set():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train / 255.0, y_train), (x_test / 255.0, y_test)


def build_model():
    model = tf.keras.models.Sequential([
        # サンプル通りだと 28*28 の配列なので、一次元配列に変換しなおす
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


def pred():
    model = build_model()
    train_tuple, test_tuple = load_data_set()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train_tuple[0].reshape(-1, 60000, 28 * 28)[0]
    model.fit(train_tuple[0], train_tuple[1], epochs=5)
    result = model.evaluate(test_tuple[0], test_tuple[1])
    print(result)


def build_model2():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def pred2():
    model = build_model()
    train_tuple, test_tuple = load_data_set()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model に渡す側が入力値を一次元にしとけば、model 側で一次元にする必要がない。
    model.fit(train_tuple[0].reshape(-1, 60000, 28 * 28)[0], train_tuple[1], epochs=5)
    result = model.evaluate(test_tuple[0], test_tuple[1])
    print(result)


pred()