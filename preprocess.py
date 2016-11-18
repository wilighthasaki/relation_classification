"""
数据的预处理工作，输入是训练集和测试集的原始输入
"""
import numpy as np
import re
import helper


def extract_sentences(file_path):
    """
    用来从原始数据集中提取句子和分类
    :param file_path:数据集路径
    :return:得到的原始句子和分类
    """
    # 句子和分类
    sentences = []
    tags = []
    with open(file_path, 'r', encoding='utf8') as raw_input:
        flag = 0
        for line in raw_input:
            if flag == 0:
                sentences.append(line.split('\"')[1])
            elif flag == 1:
                tags.append(line.rstrip())
            elif flag == 3:
                flag = 0
                continue
            flag += 1
    # sentences = np.array(sentences)
    # tags = np.array(tags)
    return [sentences, tags]


def remove_punctuations(sentences):
    processed_sentences = []
    a = '\'s'
    r = '[’!"#$%&\'()*+,-.:;=?@[\\]^_`{|}~]+'
    for line in sentences:
        line = re.sub(a, '', line)
        line = re.sub(r, '', line)
        processed_sentences.append(line.rstrip().lower())
    return processed_sentences


def apart_tags(sentences):
    left_tag = '>'
    right_tag = '<'
    blank = '[ ]+'
    new_lines = []
    for line in sentences:
        line = re.sub(left_tag, '> ', line)
        line = re.sub(right_tag, ' <', line)
        line = re.sub(blank, ' ', line)
        line = line.strip()
        new_lines.append(line)
    return new_lines


def trans4glove(sentences, output_path):
    with open(output_path, 'a', encoding='utf8') as output_data:
        for i in sentences:
            output_data.write(i + '\n')


def sentences2word(sentences, output_path):
    word_lists = []
    for line in sentences:
        words = line.strip().split()
        word_lists.append(words)
    word_lists = np.array(word_lists)
    np.save(output_path, word_lists)


def build_map(text_path="data/glove_input.txt", tag_path=''):
    """
    建立id到
    :param train_path:
    :return:
    """
    train_data = np.load('data/train_file.npy')
    test_data = np.load('data/test_file_full.npy')
    words = []
    for i in train_data:
        for j in i:
            if j not in words:
                words.append(j)
    for i in test_data:
        for j in i:
            if j not in words:
                words.append(j)

    labels = list(set(np.load('data/targets.npy')))
    word2id = dict(zip(words, range(1, len(words) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))
    id2word = dict(zip(range(1, len(words) + 1), words))
    id2label = dict(zip(range(1, len(labels) + 1), labels))
    id2word[0] = "<PAD>"
    id2label[0] = "<PAD>"
    word2id["<PAD>"] = 0
    label2id["<PAD>"] = 0
    id2word[len(words) + 1] = "<NEW>"
    word2id["<NEW>"] = len(words) + 1

    helper.save_map(id2word, id2label)

    return word2id, id2word, label2id, id2label


def process(path):
    """
    整体流程
    :param path:
    :return:
    """
    raw_sententces, targets = extract_sentences(path)
    pure_sentences = remove_punctuations(raw_sententces)
    # pure_sentences = np.array(pure_sentences)
    final_sentences = apart_tags(pure_sentences)
    # trans4glove(final_sentences, 'data/glove_input.txt')
    words_output = path.split('.')[0].lower() + '.npy'
    sentences2word(final_sentences, words_output)
    np.save('data/' + path.split('/')[1].split('_')[0].lower() + '_targets.npy', targets)


if __name__ == '__main__':
    demo_path = 'data/TRAIN_SMALL.TXT'
    test_path = 'data/TEST_FILE_FULL.TXT'
    train_path = 'data/TRAIN_FILE.TXT'
    process(train_path)
    process(test_path)
    # build_map()