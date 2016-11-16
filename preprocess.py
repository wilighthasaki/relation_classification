"""
数据的预处理工作，输入是训练集和测试集的原始输入
"""
import numpy as np
import re


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
        processed_sentences.append(line.rstrip())
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
        new_lines.append(line)
    return new_lines


def trans4glove(sentences, output_path):
    with open(output_path, 'a', encoding='utf8') as output_data:
        for i in sentences:
            output_data.write(i + '\n')


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


if __name__ == '__main__':
    demo_path = 'data/TRAIN_SMALL.TXT'
    test_path = 'data/TEST_FILE_FULL.TXT'
    train_path = 'data/TRAIN_FILE.TXT'
    process(train_path)
    process(test_path)