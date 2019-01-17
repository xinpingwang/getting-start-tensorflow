import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据加载，如果 data_set/mnist 目录下没有对应的数据，将自动下载
mnist = input_data.read_data_sets("data_set/mnist", one_hot=True)

# mnist 数据集中的图片为 28x28
x = tf.placeholder(tf.float32, [None, 784])
# 标签值 [0...9]
y = tf.placeholder(tf.float32, [None, 10])

# 要训练的权重数据
W = tf.Variable(tf.zeros([784, 10]))
# 要训练的偏差
b = tf.Variable(tf.zeros([10]))

# 模型：全联接后面加 softmax 函数
y_ = tf.nn.softmax(tf.matmul(x, W) + b)

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
    for _ in range(1000):
        # 每次取 100 个进行训练
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

    # 测试模型在测试集上的准确率
    result = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print(result)
