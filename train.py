import helper
import argparse
import tensorflow as tf
import numpy as np
from BLSTM_ATT import BLSTM_ATT

# python train.py train.in model -v validation.in -c char_emb -e 10 -g 2

parser = argparse.ArgumentParser()
parser.add_argument("train_path", help="the path of the train file")
parser.add_argument("label_path", help="the path of the label file")
parser.add_argument("save_path", help="the path of the saved model")
parser.add_argument("-e","--epoch", help="the number of epoch", default=100, type=int)
parser.add_argument("-w","--word_emb", help="the char embedding file", default=None)
parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)


args = parser.parse_args()

train_path = args.train_path
label_path = args.label_path
save_path = args.save_path
num_epochs = args.epoch
emb_path = args.word_emb
gpu_config = "/gpu:"+str(args.gpu)
num_steps = 100

char2id, id2char = helper.load_map("word2id")
label2id, id2label = helper.load_map("label2id")

X_train = np.load(train_path)
X_train = np.array([[char2id[word] for word in line] for line in X_train])
X_train = helper.padding(X_train, 100)

y_train = np.load(label_path)
y_train = np.array([label2id[label] for label in y_train])

# 正在读取词向量
embedding_matrix = helper.get_embedding(emb_path)


print("building model")
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = BLSTM_ATT(50, 100, embedding_matrix)
        print("training model")
        tf.initialize_all_variables().run()
        model.train(sess, save_path, X_train, y_train)

