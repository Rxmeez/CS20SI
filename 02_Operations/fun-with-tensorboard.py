import tensorflow as tf

# x = a + b
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

# To see the names of each node on Tensorboard you have to
# explicitly name them

c = tf.constant(3, name='c')
d = tf.constant(4, name='d')
y = tf.add(c, d, name='y-Add')

e = tf.Variable(2, name='e')
f = tf.Variable(3, name='f')
z = tf.add(e, f, name='z-Add')

# Initialize Variables
init = tf.global_variables_initializer()
# in tf.Session() ==> tf.run(init)

# To initialize only some values it would be written like:
init_ef = tf.variables_initializer([e, f], name='init_ef')
# in tf.Session() ==> tf.run(init_ef)

# To activate TensorBoard on this program, add the following line
# after you built your graph right before running the training loop.
#   writer = tf.summary.FileWriter(logs_dir, sess.graph)
# Creates a writer object to write operations to the event file, stored
# in the folder logs_dir. You can choose logs_dir to be something
# such as './graphs'.

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(init)
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(z))

# Close the writer when you're done using it
writer.close()
