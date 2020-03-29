import tensorflow as tf

import numpy as np



import plotting

def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped

def main():

    #Dataset
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.boston_housing.load_data()
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32).reshape(-1,1)

    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32).reshape(-1,1)

    #Dataset Variables 
    train_size, num_features = x_train.shape
    _ , num_target = y_train.shape
    test_size, _ = x_test.shape

    #Model variables
    epochs = 4400
    lr = 0.001
    bs = 512
    initW =tf.keras.initializers.GlorotUniform()
    initB = tf.keras.initializers.Zeros()

    hiddL = 200
                    

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = hiddL, input_shape=(num_features,),kernel_initializer = initW, bias_initializer = initB))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dense(units = num_target, input_shape=(num_features,)))
    model.summary()

    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(learning_rate =lr),
        metrics = [r_squared]
    )

    model.fit(
        x = x_train,
        y = y_train,
        epochs = epochs,
        batch_size = bs,
        validation_data = [x_test,y_test]
    )

    score = model.evaluate(x_test,y_test)
    print(score)
    



if __name__ == "__main__":
    main()