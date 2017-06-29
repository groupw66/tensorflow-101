import tensorflow as tf
import data

train_points, train_labels, test_points, test_labels = data.gen_circle_train_test_data()
# data.plot(train_points, train_labels)
# data.plot(test_points, test_labels)
train_labels = train_labels.reshape([-1, 1])
test_labels = test_labels.reshape([-1, 1])

# input feature
X = tf.placeholder(tf.float32, [None, 2])
Y_label = tf.placeholder(tf.float32, [None, 1])

# hidden layer 1
L1_size = 3
W1 = tf.Variable(tf.random_uniform([2, L1_size], -1, 1, seed=0))
B1 = tf.Variable(tf.zeros([L1_size]))
Y1 = tf.nn.relu(X @ W1 + B1)

# last layer (one perceptron)
W_last = tf.Variable(tf.zeros([L1_size, 1]))
B_last = tf.Variable(tf.zeros([1]))
Y_predict = tf.tanh(Y1 @ W_last + B_last)

error = tf.subtract(Y_label, Y_predict)
mse = tf.reduce_mean(tf.square(error))

train = tf.train.GradientDescentOptimizer(0.03).minimize(mse)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train
err, target = 1., 0.000001
epoch, max_epochs = 0, 5000
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train], feed_dict={X: train_points, Y_label: train_labels})
    if epoch % 100 == 0:
        print('epoch:', epoch, 'mse:', err)
print('epoch:', epoch, 'mse:', err)

# validate
err_test = sess.run([mse], feed_dict={X: test_points, Y_label: test_labels})
print('mse test:', err_test)
