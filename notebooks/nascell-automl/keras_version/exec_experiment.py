import sys
import argparse

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # show only errors 

import tensorflow as tf

from tensorflow.keras.datasets import mnist

from cnn import CNN_Keras


def main(action, name):
    (train_x,train_y), (test_x,test_y) = mnist.load_data()
    print(f'train.shape: {train_x.shape},{train_y.shape}')
    print(f'test.shape: {test_x.shape},{test_y.shape}')

    # normalize
    train_x, test_x = train_x / 255., test_x/255.
    
    train_x = tf.expand_dims(train_x, axis=0)
    train_y = tf.expand_dims(train_y, axis=0)
    test_x = tf.expand_dims(test_x, axis=0)
    test_y = tf.expand_dims(test_y, axis=0)
    
    action = [int(x) for x in action.split(",")]
    training_epochs = 10 
    batch_size = 100

    action = [action[x:x+4] for x in range(0, len(action), 4)]
    cnn_drop_rate = [c[3] for c in action]
    
    print(f'action: {action}')
    print(f'cnn_drop_rate: {cnn_drop_rate}')
    
    cnn_keras = CNN_Keras(10, action, cnn_drop_rate, batch_size)
    cnn_keras.build_model()
    cnn_keras.compile_model()
    
    model = cnn_keras.model
    
    print(f'train.shape: {train_x.shape},{train_y.shape}')
    print(f'test.shape: {test_x.shape},{test_y.shape}')
    
    print(f'name: {name}')
    print(f'action: {action}')
    
    H = model.fit(train_x, 
                  train_y, 
                  epochs=training_epochs,
                  batch_size=batch_size)
    
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
    
    test_acc, test_loss = model.evaluate(test_x, test_y)
    print(f'Test Accuracy: {test_acc}')
    print(f'Test Loss: {test_loss}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', default="5, 32, 2,  5, 3, 64, 2, 3")
    parser.add_argument('--name', default="model")
    args = parser.parse_args()

    main(args.architecture, args.name)

