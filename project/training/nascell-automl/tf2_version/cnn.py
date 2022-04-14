import tensorflow as tf

class CNN():
    def __init__(self, num_input, num_classes, cnn_config):
        cnn = [c[0] for c in cnn_config]
        cnn_num_filters = [c[1] for c in cnn_config]
        max_pool_ksize = [c[2] for c in cnn_config]

        self.X = tf.compat.v1.placeholder(tf.float32,
                                [None, num_input], 
                                name="input_X")
        self.Y = tf.compat.v1.placeholder(tf.int32, [None, num_classes], name="input_Y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, [], name="dense_dropout_keep_prob")
        self.cnn_dropout_rates = tf.compat.v1.placeholder(tf.float32, [len(cnn), ], name="cnn_dropout_keep_prob")

        Y = self.Y
        X = tf.expand_dims(self.X, -1)
        pool_out = X
        with tf.compat.v1.name_scope("Conv_part"):
            for idd, filter_size in enumerate(cnn):
                with tf.compat.v1.name_scope("L"+str(idd)):
                    conv_out = tf.compat.v1.layers.conv1d(
                        pool_out,
                        filters=cnn_num_filters[idd],
                        kernel_size=(int(filter_size)),
                        strides=1,
                        padding="SAME",
                        name="conv_out_"+str(idd),
                        activation=tf.nn.relu,
                        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                        bias_initializer=tf.compat.v1.zeros_initializer
                    )
                    pool_out = tf.compat.v1.layers.max_pooling1d(
                        conv_out,
                        pool_size=(int(max_pool_ksize[idd])),
                        strides=1,
                        padding='SAME',
                        name="max_pool_"+str(idd)
                    )
                    pool_out = tf.nn.dropout(pool_out, 1 - (self.cnn_dropout_rates[idd]))

            flatten_pred_out = tf.contrib.layers.flatten(pool_out)
            self.logits = tf.compat.v1.layers.dense(flatten_pred_out, num_classes)

        self.prediction = tf.nn.softmax(self.logits, name="prediction")
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.stop_gradient(Y), name="loss")
        correct_pred = tf.equal(tf.argmax(input=self.prediction, axis=1), tf.argmax(input=Y, axis=1))
        self.accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_pred, tf.float32), name="accuracy")
