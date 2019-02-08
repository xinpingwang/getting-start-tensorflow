import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据加载，如果 data_set/mnist 目录下没有对应的数据，将自动下载
mnist = input_data.read_data_sets("data_set/mnist", one_hot=True)

# mnist 数据集中的图片为 28x28
x = tf.placeholder(tf.float32, [None, 784])
# 标签值 [0...9]
y = tf.placeholder(tf.float32, [None, 10])

# 开始计算
# 1、将输入数据转化为 conv2d 可以接受的数据
reshape_x = tf.reshape(tensor=x, shape=[-1, 28, 28, 1])
# 2、使用 6 个 5x5 的卷积核分别对扩展后的图片进行卷积操作，得到 28x28x6 的输出
conv1 = tf.layers.conv2d(inputs=reshape_x, filters=6, kernel_size=5, padding='same', activation=tf.nn.sigmoid)
# 3、平均池化，得到 14x14x6 的输出
p2 = tf.layers.average_pooling2d(inputs=conv1, pool_size=2, strides=2)
# 4、使用 16 个 5x5x6 的卷积核对池化后的图像进行卷积操作，得到 10x10x16 的输出
conv3 = tf.layers.conv2d(p2, 16, 5, activation=tf.nn.sigmoid)
# 5、平均池化，得到 5x5x16 的输出
p4 = tf.layers.average_pooling2d(conv3, 2, 2)
# 6、使用 120 个 5x5x16 的卷积核对池化后的图像进行卷积操作，得到 1x1x120 的输出
conv5 = tf.layers.conv2d(p4, 120, 5, activation=tf.nn.sigmoid)
# 7、全连接，输出 84
f6 = tf.layers.dense(conv5, 84, activation=tf.nn.sigmoid)
# 8、全联接，输出 10
f7 = tf.layers.dense(f6, 10, activation=tf.nn.sigmoid)
# 将输出转换为与标签一致的 shape ((?, 1, 1, 10) -> (?, 10))
y_ = tf.reshape(f7, [-1, 10])

# 交叉熵损失函数
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))

# 训练操作 (使用上面定义的损失函数对训练参数进行优化，学习速率为 0.5)
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初始话变量操作
init_op = tf.global_variables_initializer()

# 统计预测结果和标签相同的数量
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 运行初始话
    init_op.run()
    # 训练 1000 次
    for _ in range(2000):
        # 每次取 100 个进行训练
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # batch_xs = np.reshape(batch_xs, (28, 28))
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

    # 测试模型在测试集上的准确率
    result = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print(result)
