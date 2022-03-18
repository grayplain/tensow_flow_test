import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import tensorflow.keras.layers as layers


def load_dataset():
    boston = load_boston()
    return boston.data, boston.target

def build_model():
    return tf.keras.models.Sequential([
        layers.Dense(13),
        # 中間層のニューロンの数は適当。
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.1),
        # 回帰はこれでいいのか？
        layers.Dense(1)
    ])


def main():
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    print(y_test)
    return
    model =build_model()
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])


    model.fit(x_train, y_train, epochs=10)
    result = model.evaluate(x_test, y_test)
    print(result)
    print("--------")
    predict_value = model.predict(x_test)
    print(predict_value)


main()