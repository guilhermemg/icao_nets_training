import tensorflow as tf
import argparse
import sys
from cnn import CNN
#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow_datasets as tfds
#from tf.keras.datasets.mnist import load_data

def main(action, name):
    #mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    (ds_train, ds_test), df_info = tfds.load(name="mnist", split=['train','test'], as_supervised=True, with_info=True)
    #mnist_train, mnist_test = load_data()
    #train, test = tf.keras.datasets.mnist.load_data()
    print(ds_train.as_numpy().shape, test.as_numpy().shape)
    
    action = [int(x) for x in action.split(",")]
    training_epochs = 10 
    batch_size = 100

    action = [action[x:x+4] for x in range(0, len(action), 4)]
    cnn_drop_rate = [c[3] for c in action]

    model = CNN(784, 10, action)
    loss_op = tf.reduce_mean(input_tensor=model.loss)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss_op)
    
    tf.compat.v1.summary.scalar('acc', model.accuracy)
    tf.compat.v1.summary.scalar('loss', tf.reduce_mean(input_tensor=model.loss))
    merged_summary_op = tf.compat.v1.summary.merge_all()
    summary_writer = tf.compat.v1.summary.FileWriter(name, graph=tf.compat.v1.get_default_graph())

    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)

    
    for epoch in range(training_epochs):
        #for step in range(int(mnist.train.num_examples/batch_size)):
        for step, batch_x, batch_y in range(int(len(ds_train)/batch_size)), tfds.as_numpy(ds_train):
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x, batch_y = train.take(batch_size).as_numpy_iterator()
            print(batch_x.shape, batch_y.shape)
            feed = {model.X: batch_x,
                    model.Y: batch_y,
                    model.dropout_keep_prob: 0.85,
                    model.cnn_dropout_rates: cnn_drop_rate}
            _, summary = sess.run([train_op, merged_summary_op], feed_dict=feed)
            summary_writer.add_summary(summary, step+(epoch+1)*int(mnist.train.num_examples/batch_size))

        print("epoch: ", epoch+1, " of ", training_epochs)
    
        #batch_x, batch_y = mnist.test.next_batch(mnist.test.num_examples)
        batch_x, batch_y = test.take(len(test)).as_numpy_iterator()
        loss, acc = sess.run(
                               [loss_op, model.accuracy],
                               feed_dict={model.X: batch_x,
                                          model.Y: batch_y,
                                          model.dropout_keep_prob: 1.0,
                                          model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
       
        print("Network accuracy =", acc, " loss =", loss)
    print("Final accuracy for", name, " =", acc)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', default="5, 32, 2,  5, 3, 64, 2, 3")
    parser.add_argument('--name', default="model")
    args = parser.parse_args()

    main(args.architecture, args.name)

