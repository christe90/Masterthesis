class RelativeDifferenceLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(RelativeDifferenceLoss, self).__init__(name='RelativeDifferencePercentage')

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        lossValue = (tf.math.abs(y_pred - y_true) / tf.math.maximum(tf.math.abs(y_pred), tf.math.abs(y_true)))
        return tf.reduce_mean(lossValue*100, axis = -1)
