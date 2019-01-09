import tensorflow as tf

# 定义一个常量
constant = tf.constant(3.0)

# 定义一个变量, 变量在使用前必须进行初始话操作，line 14， 15
var = tf.Variable(2.0)

# 定义一个乘法操作
multiply_op = tf.multiply(constant, var)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    # var.initializer.run()
    sess.run(init_op)
    result = sess.run(multiply_op)
    print(result)
