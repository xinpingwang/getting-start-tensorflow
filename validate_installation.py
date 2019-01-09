import tensorflow as tf

# 构造计算节点
hello = tf.constant("Hello, TensorFlow")
# # 运行上下文
# sess = tf.Session()
# # 运行
# result = sess.run(hello)
# print(result)
# # 释放资源
# sess.close()

with tf.Session() as sess:
    result = sess.run(hello)
    print(result)
