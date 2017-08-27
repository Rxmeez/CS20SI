import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
import xlrd

DATA_FILE = 'slr05.xls'
# Import Data
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

df = pd.read_excel(DATA_FILE)

# Create placeholders for input X (number of fire) and label (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Create weight and bias
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# Y = wX + b - construct model to predict
y_predicted = w * X + b

# Square error - loss function
loss = tf.square(Y - y_predicted, name='loss')

# Gradient descent with learning rate 0.01 - minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        print("Epoch {}".format(i + 1))
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})

    w_value, b_value = sess.run([w, b])

print(w_value, b_value)
print(l)

y_p = w_value * df['X'] + b_value
plt.scatter(df['X'], df['Y'])
plt.plot(df['X'], y_p, color='r')
plt.show()
