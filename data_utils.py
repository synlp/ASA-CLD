import os
import re
import logging
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, aspect, seg_list=[], label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.seg_list = seg_list
        self.aspect = aspect
        self.label = label

class DiaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir, tokenizer, max_seq_len=512, batch_size=16):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size


    def get_train_examples(self, dataset):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, dataset, "train.txt")), "train")

    def get_test_examples(self, dataset):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, dataset, "test.txt")), "test")

    def get_dev_examples(self, dataset):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, dataset, "dev.txt")), "dev")

    def get_labels(self):
        return ["-1","0","1"]

    def get_label_map(self):
        label_map = {label: i for i, label in enumerate(self.get_labels())}
        return label_map

    def get_id2label_map(self):
        return {i: label for label, i in self.get_label_map().items()}

    def get_tag_size(self):
        return len(self.get_labels())

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, seg_list, aspect, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text=sentence, aspect=aspect, seg_list=seg_list, label=label))
        return examples

    def _read_txt(self, file_path):
        '''
        read file
        return format :
        '''
        datas = []
        sentence_list = []
        with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            for i, line in enumerate(fin):
                if i % 3 == 0:
                    datas.append([])
                datas[-1].append(line.strip())
            for data in datas:
                seg_list = re.split("\$T\$", data[0])
                aspect = data[1]
                polarity = data[2]
                aspect = "<e1> {} </e1>".format(aspect)
                sentence = aspect.join(seg_list)
                sentence_list.append([sentence, seg_list, aspect, polarity])

        return sentence_list

    def convert_examples_to_features(self, tokenizer, examples, max_seq_length):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = self.get_label_map()
        label_map = {"0":0, "1":1, "-1":2}

        features = []
        for (ex_index, example) in enumerate(examples):
            text = example.text
            label = example.label
            aspect = example.aspect
            tokens = text.split(' ')
            ntokens = ["[CLS]"]
            aspect_value = 0
            aspect_mask = [0]
            for word in tokens:
                if word in ["<e1>", "</e1>"]:
                    if word in ["<e1>"]:
                        aspect_value = 1
                    if word in ["</e1>"]:
                        aspect_value = 0
                    continue
                token = tokenizer.tokenize(word)
                if len(ntokens) + len(token) > max_seq_length - 1:
                    break
                ntokens.extend(token)
                for i in range(len(token)):
                    aspect_mask.append(aspect_value)
    
            ntokens.append("[SEP]")
            aspect_mask.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            label_id = label_map[label]
            aspect_mask += [0] * (max_seq_length - len(aspect_mask))

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(aspect_mask) == max_seq_length

            features.append({
                "input_ids":torch.tensor(input_ids, dtype=torch.long),
                "aspect_mask":torch.tensor(aspect_mask, dtype=torch.long),
                "input_mask":torch.tensor(input_mask, dtype=torch.long),
                "segment_ids":torch.tensor(segment_ids, dtype=torch.long),
                "label_ids":torch.tensor(label_id, dtype=torch.long)
            })
        return features

    def _get_dataloader(self, features, batch_size, mode='train', rank=0, world_size=1):
        if mode == "train" and world_size > 1:
            features = features[rank::world_size]

        data_set = DiaDataset(features)
        sampler = RandomSampler(data_set)
        return DataLoader(data_set, sampler=sampler, batch_size=batch_size)

    def get_all_train_dataloader(self, dataset_list):
        tokenizer = self.tokenizer
        train_examples = []
        for dataset in dataset_list:
            train_examples.extend(self.get_train_examples(dataset))
        train_features = self.convert_examples_to_features(tokenizer, train_examples, self.max_seq_len)
        train_dataloader = self._get_dataloader(train_features, mode="train", batch_size=self.batch_size)
        return train_dataloader
    
    def get_train_dataloader(self, dataset):
        tokenizer = self.tokenizer
        train_examples = self.get_train_examples(dataset)
        train_features = self.convert_examples_to_features(tokenizer, train_examples, self.max_seq_len)
        train_dataloader = self._get_dataloader(train_features, mode="train", batch_size=self.batch_size)
        return train_dataloader

    def get_test_dataloader(self, dataset):
        tokenizer = self.tokenizer
        test_examples = self.get_test_examples(dataset)
        test_features = self.convert_examples_to_features(tokenizer, test_examples, self.max_seq_len)
        test_dataloader = self._get_dataloader(test_features, mode="test", batch_size=self.batch_size)
        return test_dataloader

    def get_dev_dataloader(self, dataset):
        tokenizer = self.tokenizer
        dev_examples = self.get_dev_examples(dataset)
        dev_features = self.convert_examples_to_features(tokenizer, dev_examples, self.max_seq_len)
        dev_dataloader = self._get_dataloader(dev_features, mode="dev", batch_size=self.batch_size)
        return dev_dataloader

    def get_dataloader(self, dataset):
        tokenizer = self.tokenizer
        #train
        train_examples = self.get_train_examples(dataset)
        train_features = self.convert_examples_to_features(tokenizer, train_examples, self.max_seq_len)
        train_dataloader = self._get_dataloader(train_features, mode="train", batch_size=self.batch_size)

        # test
        test_examples = self.get_test_examples(dataset)
        test_features = self.convert_examples_to_features(tokenizer, test_examples, self.max_seq_len)
        test_dataloader = self._get_dataloader(test_features, mode="test", batch_size=self.batch_size)

        # dev
        dev_examples = self.get_dev_examples(dataset)
        dev_features = self.convert_examples_to_features(tokenizer, dev_examples, self.max_seq_len)
        dev_dataloader = self._get_dataloader(dev_features, mode="dev", batch_size=self.batch_size)
        
        return train_dataloader, test_dataloader, dev_dataloader

def seq_reduce_ret_len(batch_data, pad=0):
    batch_size, seq_len = batch_data.shape
    max_seq_len = seq_len
    while max_seq_len > 1:
        if (batch_data[:,max_seq_len-1] != pad).sum().item() > 0:
            break
        max_seq_len -= 1
    return max_seq_len