import os
import csv
import numpy as np
import pandas as pd


def get_embedding(infile_path="embedding", dim=50):
    char2id, id_char = load_map("word2id")
    # 默认为50维
    emb_matrix = np.zeros((len(char2id), dim))
    with open(infile_path, "r") as infile:
        for row in infile:
            row = row.strip()
            items = row.split()
            char = items[0]
            emb_vec = np.array([float(val) for val in items[1:]])
            if char in char2id.keys():
                emb_matrix[char2id[char]] = emb_vec

    for i in range(1, len(char2id)):
        if emb_matrix[i][1] == 0:
            emb_matrix[i] = np.random.rand(50)
    return emb_matrix


def next_batch(x, y, start_index, batch_size=128):
    last_index = start_index + batch_size
    x_batch = list(x[start_index:min(last_index, len(x))])
    y_batch = list(y[start_index:min(last_index, len(x))])
    if last_index > len(x):
        left_size = last_index - (len(x))
        for i in range(left_size):
            index = np.random.randint(len(x))
            x_batch.append(x[index])
            y_batch.append(y[index])
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def next_random_batch(x, y, batch_size=128):
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(x))
        x_batch.append(x[index])
        y_batch.append(y[index])
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


# use "0" to padding the sentence
def padding(sample, seq_max_len):
    new_list = []
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
        new_list.append(sample[i])
    return np.array(new_list)


def prepare(chars, labels, seq_max_len, is_padding=True):
    x = []
    y = []
    tmp_x = []
    tmp_y = []

    for record in zip(chars, labels):
        c = record[0]
        l = record[1]
        # empty line
        if c == -1:
            if len(tmp_x) <= seq_max_len:
                x.append(tmp_x)
                y.append(tmp_y)
            tmp_x = []
            tmp_y = []
        else:
            tmp_x.append(c)
            tmp_y.append(l)
    if is_padding:
        x = np.array(padding(x, seq_max_len))
    else:
        x = np.array(x)
    y = np.array(padding(y, seq_max_len))

    return x, y


def load_map(token2id_filepath):
    token2id = {}
    id2token = {}
    with open(token2id_filepath) as infile:
        for row in infile:
            row = row.rstrip()
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token


def save_map(id2word, id2label):
    with open("word2id", "w") as outfile:
        for idx in id2word:
            outfile.write(id2word[idx] + "\t" + str(idx) + "\r\n")
    with open("label2id", "w") as outfile:
        for idx in id2label:
            outfile.write(id2label[idx] + "\t" + str(idx) + "\r\n")
    print("saved map between token and id")


def get_train(train_path, label_path, seq_max_len=100):

    word2id, id2word = load_map("word2id")
    label2id, id2label = load_map("label2id")

    train_data = np.load(train_path)
    labels = np.load(label_path)

    # map the word and label into id
    df_train["word_id"] = df_train.word.map(lambda x: -1 if str(x) == str(np.nan) else word2id[x])
    df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else label2id[x])

    # convert the data in maxtrix
    x, y = prepare(df_train["word_id"], df_train["label_id"], seq_max_len)

    # shuffle the samples
    num_samples = len(x)
    indexs = np.arange(num_samples)
    np.random.shuffle(indexs)
    x = x[indexs]
    y = y[indexs]

    if val_path != None:
        x_train = x
        y_train = y
        x_val, y_val = get_test(val_path, is_validation=True, seq_max_len=seq_max_len)
    else:
        # split the data into train and validation set
        x_train = x[:int(num_samples * train_val_ratio)]
        y_train = y[:int(num_samples * train_val_ratio)]
        x_val = x[int(num_samples * train_val_ratio):]
        y_val = y[int(num_samples * train_val_ratio):]

    print("train size: %d, validation size: %d" %(len(x_train), len(y_val)))

    return x_train, y_train, x_val, y_val


def get_test(test_path="test.in", is_validation=False, seq_max_len=200):
    word2id, id2word = load_map("word2id")
    label2id, id2label = load_map("label2id")

    df_test = pd.read_csv(test_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None, names=["word", "label"])

    def mapFunc(x, word2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in word2id:
            return word2id["<NEW>"]
        else:
            return word2id[x]

    df_test["word_id"] = df_test.word.map(lambda x:mapFunc(x, word2id))
    df_test["label_id"] = df_test.label.map(lambda x : -1 if str(x) == str(np.nan) else label2id[x])

    if is_validation:
        x_test, y_test = prepare(df_test["word_id"], df_test["label_id"], seq_max_len)
        return x_test, y_test
    else:
        df_test["word"] = df_test.word.map(lambda x : -1 if str(x) == str(np.nan) else x)
        x_test, _ = prepare(df_test["word_id"], df_test["word_id"], seq_max_len)
        x_test_str, _ = prepare(df_test["word"], df_test["word_id"], seq_max_len, is_padding=False)
        print("test size: %d" %(len(x_test)))
        return x_test, x_test_str
