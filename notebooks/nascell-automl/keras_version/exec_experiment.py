import sys
import argparse

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors 

import tensorflow as tf

from tensorflow.keras.datasets import mnist

from cnn import CNN_Keras


def main(action, name):
    batch_size = 100
    action = [int(x) for x in action.split(",")]
    training_epochs = 10 

    action = [action[x:x+4] for x in range(0, len(action), 4)]
    cnn_drop_rate = [c[3] for c in action]
    
    print(f'batch_size: {batch_size}')
    print(f'training_epochs: {training_epochs}')
    print(f'action: {action}')
    print(f'cnn_drop_rate: {cnn_drop_rate}')
    print(f'name: {name}')
    
    (X_train,y_train), (X_test,y_test) = tf.keras.datasets.mnist.load_data()

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)

    X_train_ds = tf.data.Dataset.from_tensor_slices(X_train)
    X_test_ds = tf.data.Dataset.from_tensor_slices(X_test)
    
    X_train_ds = X_train_ds.map(lambda x: x/255.)
    X_test_ds = X_test_ds.map(lambda x: x/255.)
    
    y_train_ds = tf.data.Dataset.from_tensor_slices(y_train)
    y_test_ds = tf.data.Dataset.from_tensor_slices(y_test)
    
    train_ds = tf.data.Dataset.zip((X_train_ds, y_train_ds))
    test_ds = tf.data.Dataset.zip((X_test_ds, y_test_ds))

    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    
    
    cnn_keras = CNN_Keras(10, action, cnn_drop_rate, batch_size)
    cnn_keras.build_model()
    cnn_keras.compile_model()
    
    model = cnn_keras.model
    
    H = model.fit(train_ds, 
                  epochs=training_epochs)
    
    #for epoch in range(training_epochs):
    #    for step in range(int(mnist.train.num_examples/batch_size)):
    #        batch_x, batch_y = mnist.train.next_batch(batch_size)
    #        print(batch_x.shape, batch_y.shape)
    #        feed = {model.X: batch_x,
    #                model.Y: batch_y,
    #                model.dropout_keep_prob: 0.85,
    #                model.cnn_dropout_rates: cnn_drop_rate}
    #        _, summary = sess.run([train_op, merged_summary_op], feed_dict=feed)
    #
    #    print("epoch: ", epoch+1, " of ", training_epochs)
    #
    #    batch_x, batch_y = mnist.test.next_batch(mnist.test.num_examples)
    #    loss, acc = sess.run(
    #                           [loss_op, model.accuracy],
    #                           feed_dict={model.X: batch_x,
    #                                      model.Y: batch_y,
    #                                      model.dropout_keep_prob: 1.0,
    #                                      model.cnn_dropout_rates: [1.0]*len(cnn_drop_rate)})
    #   
    #    print("Network accuracy =", acc, " loss =", loss)
    
    test_loss,test_acc = model.evaluate(test_ds)
    print(f'Test Accuracy: {round(test_acc*100,2)}%')
    print(f'Test Loss: {round(test_loss,4)}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', default="5, 32, 2,  5, 3, 64, 2, 3")
    parser.add_argument('--name', default="model")
    args = parser.parse_args()

    main(args.architecture, args.name)

