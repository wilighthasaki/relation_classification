import math
import helper
import numpy as np
import tensorflow as tf
import pickle


class BLSTM_ATT(object):
    """
    参考论文中的基于BLSTM-Attention模型来搭建。
    """
    def __init__(self, input_dim, num_steps, embedding_matrix, num_classes=19, is_training=True, num_epochs=20,
                 batch_size=32, hidden_dim=100, learning_rate=0.005, dropout=0):
        # 初始化参数
        self.input_dim = input_dim
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.dropout = dropout

        # 输入输出
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None])

        # 转化输入
        self.input_emb = tf.nn.embedding_lookup(embedding_matrix, self.inputs)
        self.input_emb = tf.transpose(self.input_emb, [1, 0, 2])
        self.input_emb = tf.reshape(self.input_emb, [-1, self.input_dim])
        self.input_emb = tf.split(0, self.num_steps, self.input_emb)
        self.input_shape = tf.shape(self.input_emb)
        # 双向LSTM层
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        # 如果是在训练就dropout
        if is_training:
            lstm_cell_fw =\
                tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout))



    def train(self, sess, save_file, X_train, y_train):
        flag = 0
        # 用来得到id到词的映射和id到标签的映射。
        char2id, id2char = helper.load_map("word2id")
        label2id, id2label = helper.load_map("label2id")

        # 获得迭代次数，向上取整。
        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))

        cnt = 0
        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            # 打乱训练集
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]
            # 开始训练。
            print("current epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                # train
                # 获得下一批数据
                X_train_batch, y_train_batch = helper.next_batch(X_train, y_train, start_index=iteration * self.batch_size, batch_size=self.batch_size)

                results =\
                    sess.run([
                        self.input_shape
                    ],
                    feed_dict={
                        self.inputs: X_train_batch,
                        self.targets: y_train_batch,
                    })
                print(results[0])
                #
                # if iteration % 10 == 0:
                #     cnt += 1
                #     precision_train, recall_train, f1_train = self.evaluate(X_train_batch, y_train_batch, predicts_train, id2char, id2label)
                #     summary_writer_train.add_summary(train_summary, cnt)
                #     print("iteration: %5d, train loss: %5d, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (iteration, loss_train, precision_train, recall_train, f1_train))
                #     if flag < 100:
                #         flag += 1
                #         if flag == 100:
                #             pickle.dump(y_train_batch, open('train_message/y_train_batch', 'wb'))
                #             pickle.dump(X_train_batch, open('train_message/X_train_batch', 'wb'))
                #             pickle.dump(predicts_train, open('train_message/predicts_train', 'wb'))
                #             pickle.dump(id2char, open('train_message/id2char', 'wb'))
                #             pickle.dump(id2label, open('train_message/id2label', 'wb'))

    def test(self, sess, X_test, X_test_str, output_path):
        char2id, id2char = helper.loadMap("char2id")
        label2id, id2label = helper.loadMap("label2id")
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print("number of iteration: " + str(num_iterations))
        with open(output_path, "w") as outfile:
            for i in range(num_iterations):
                print("iteration: " + str(i + 1))
                results = []
                X_test_batch = X_test[i * self.batch_size : (i + 1) * self.batch_size]
                X_test_str_batch = X_test_str[i * self.batch_size : (i + 1) * self.batch_size]
                if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                    X_test_batch = list(X_test_batch)
                    X_test_str_batch = list(X_test_str_batch)
                    last_size = len(X_test_batch)
                    X_test_batch += [[0 for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_str_batch += [['x' for j in range(self.num_steps)] for i in range(self.batch_size - last_size)]
                    X_test_batch = np.array(X_test_batch)
                    X_test_str_batch = np.array(X_test_str_batch)
                    results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)
                    results = results[:last_size]
                else:
                    X_test_batch = np.array(X_test_batch)
                    results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)

                for i in range(len(results)):
                    doc = ''.join(X_test_str_batch[i])
                    outfile.write(doc + "<@>" +" ".join(results[i]) + "\n")


