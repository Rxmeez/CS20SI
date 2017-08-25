import tensorflow as tf

# x = a + b
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

# To activate TensorBoard on this program, add the following line
# after you built your graph right before running the training loop.
#   writer = tf.summary.FileWriter(logs_dir, sess.graph)
# Creates a writer object to write operations to the event file, stored
# in the folder logs_dir. You can choose logs_dir to be something
# such as './graphs'.

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))

# Close the writer when you're done using it
writer.close()
