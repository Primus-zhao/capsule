#!/usr/bin/env python
# coding=utf-8

import os
import torch
import codecs

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b

print(os.getcwd())
data = './data'
raw = 'raw'
processed_folder = 'processed'
training_file = 'training.pt'
test_file = 'test.pt'

print('Processing...')

training_set = (
    read_image_file(os.path.join(data, raw, 'train-images-idx3-ubyte')),
    read_label_file(os.path.join(data, raw, 'train-labels-idx1-ubyte'))
)
test_set = (
    read_image_file(os.path.join(data, raw, 't10k-images-idx3-ubyte')),
    read_label_file(os.path.join(data, raw, 't10k-labels-idx1-ubyte'))
)
with open(os.path.join(data, processed_folder, training_file), 'wb') as f:
    torch.save(training_set, f)
with open(os.path.join(data, processed_folder, test_file), 'wb') as f:
    torch.save(test_set, f)

print('Done!')
