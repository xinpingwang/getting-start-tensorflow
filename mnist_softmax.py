import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据加载，如果 data_set/mnist 目录下没有对应的数据，将自动下载
mnist = input_data.read_data_sets("data_set/mnist", one_hot=True)

# 分别读取训练、验证和测试数据
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

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

EPOCHS = 10
BATCH_SIZE = 100

with tf.Session() as sess:
    # 运行初始话
    sess.run(init_op)
    # 训练 EPOCHS 轮，每轮都要在所有的训练数据上训练一遍
    for i in range(EPOCHS):
        # 每次读取 BATCH_SIZE 个数据进行训练
        for offset in range(0, len(X_train), BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset: end], y_train[offset: end]
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

        validate_accuracy = sess.run(accuracy, feed_dict={x: X_validation, y: y_validation})
        print("EPOCH {}, Validate Accuracy {}".format(i + 1, validate_accuracy))

    # 测试模型在测试集上的准确率
    result = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    print()
    print("Test Accuracy {}".format(result))
