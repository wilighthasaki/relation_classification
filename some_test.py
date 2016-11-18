import numpy as np


def test1():
    train_data = np.load('data/train_file.npy')
    test_data = np.load('data/test_file_full.npy')
    train_list = []
    for i in train_data:
        for j in i:
            if j not in train_list:
                train_list.append(j)

    for i in test_data:
        for j in i:
            if j not in train_list:
                print(j)


def test2():
    train_data = np.load('data/train_file.npy')
    test_data = np.load('data/test_file_full.npy')
    max_len = 0
    for i in train_data:
        if len(i) > max_len:
            max_len = len(i)
            print(i)
    for i in test_data:
        if len(i) > max_len:
            max_len = len(i)
            print(i)
    print(max_len)


if __name__ == '__main__':
    # test1()
    test2()
