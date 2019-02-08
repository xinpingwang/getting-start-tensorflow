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
# 将输入数据转化为 conv2d 可以接受的数据
reshape_x = tf.reshape(tensor=x, shape=[-1, 28, 28, 1])
# 标签值 [0...9]
y = tf.placeholder(tf.float32, [None, 10])

# 开始计算
# 第一层：卷积（输入：28x28x1，输出：28x28x6）
conv1_w = tf.Variable(tf.random_normal(shape=[5, 5, 1, 6]))
conv1_b = tf.Variable(tf.zeros(6))
# 在 LeNet 中需要的输入为 32x32，而 mnist 数据集中的数据为 28x28，这里使用 padding="SAME"，使得第一个卷积层输出跟 LeNet 中的一致
conv1 = tf.nn.conv2d(input=reshape_x, filter=conv1_w, strides=[1, 1, 1, 1], padding="SAME") + conv1_b
conv1 = tf.nn.sigmoid(conv1)
# 第二层：平均池化（输入：28x28x6，输出 14x14x6）
p2 = tf.nn.avg_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
# 第三层：卷积（输入：14x14x6，输出：10x10x16）
conv3_w = tf.Variable(tf.random_normal(shape=[5, 5, 6, 16]))
conv3_b = tf.Variable(tf.zeros(16))
conv3 = tf.nn.conv2d(input=p2, filter=conv3_w, strides=[1, 1, 1, 1], padding="VALID") + conv3_b
conv3 = tf.nn.sigmoid(conv3)
# 第四层：平均池化（输入：10x10x16，输出：5x5x16）
p4 = tf.nn.avg_pool(value=conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
# 第五层：卷积层（输入：5x5x16，输出：1x1x120）
conv5_w = tf.Variable(tf.random_normal(shape=[5, 5, 16, 120]))
conv5_b = tf.Variable(tf.zeros(120))
conv5 = tf.nn.conv2d(input=p4, filter=conv5_w, strides=[1, 1, 1, 1], padding="VALID") + conv5_b
conv5 = tf.nn.sigmoid(conv5)
# 将第五层的输出转化为 120 的数组
conv5 = tf.layers.flatten(conv5)
# 第六层：全连接（输入：120，输出：84）
f6_w = tf.Variable(tf.random_normal(shape=[120, 84]))
f6_b = tf.Variable(tf.zeros([84]))
f6 = tf.nn.sigmoid(tf.matmul(conv5, f6_w) + f6_b)
# 第七层：全联接（输入：84， 输出：10）
f7_w = tf.Variable(tf.random_normal(shape=[84, 10]))
f7_b = tf.Variable(tf.zeros([10]))
f7 = tf.nn.sigmoid(tf.matmul(f6, f7_w) + f7_b)
# 第七层输出即为网络输出
y_ = f7

# 交叉熵损失函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)
loss = tf.reduce_mean(cross_entropy)

# 训练操作 (使用上面定义的损失函数对训练参数进行优化，学习速率为 0.001)
# train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

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
