from tensorflow import keras

import time
import os

class BostonModel():

    def __init__(self, numFeatures, numTargets):
        super().__init__()

        #Model variables
        initW = keras.initializers.GlorotUniform()
        initB = keras.initializers.Zeros()
        lr = 0.001

        hiddL = 200

        #Structure
        self.m = keras.Sequential()
        self.m.add(keras.layers.Flatten())
        self.m.add(keras.layers.Dense(units = hiddL, input_shape=(numFeatures,),kernel_initializer = initW, bias_initializer = initB))
        self.m.add(keras.layers.Activation("relu"))
        self.m.add(keras.layers.Dense(units = numTargets))

        self.m.build((None,13))

        self.m.compile(
            loss = keras.losses.MeanSquaredError(),
            optimizer = keras.optimizers.Adam(learning_rate =lr),
            metrics = [keras.metrics.MSE]
        )

        

    def fit(self, *args, **kwarg):
        self.m.fit(*args, **kwarg)

    def evaluate(self, *args, **kwargs):
        return self.m.evaluate(*args,**kwargs)

    def saveWeights(self, path = "models/", name = None):
        #TODO maybe saveFileDialog
        if name is None:
            name = "model[{}].h5".format(time.time())
        
        if not os.path.exists(path):
            os.mkdir(path)

        modelFilePath = os.path.join(path,name)
        
        self.m.save_weights(filepath=modelFilePath)

    def loadWeights(self,filePath):
        self.m.load_weights(filepath = filePath)

