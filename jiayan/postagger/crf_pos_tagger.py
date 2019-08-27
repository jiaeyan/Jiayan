import random
from itertools import chain
from string import ascii_uppercase

import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from jiayan.globals import re_zh_exclude


class CRFPOSTagger:

    def __init__(self, lm):
        self.lm = lm
        self.tagger = None

    def load(self, crf_model):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(crf_model)

    def sent2features(self, sent):
        length = len(sent)
        feat_list = []
        for i, word in enumerate(sent):
            # pattern = self.get_word_pattern(word)
            # is_zh = '1' if re_zh_exclude.match(word) else '0'
            features = [
                'bias',
                '0:word=' + word,
                # '0:pattern=' + pattern,
                # '0:type=' + is_zh,
            ]

            if i > 0:
                features.extend([
                    '-1:word=' + sent[i - 1],
                    '-10:words=' + '|'.join(sent[i - 1: i + 1]),
                ])
            else:
                features.append('BOS')

            if i > 1:
                features.extend([
                    '-2:word=' + sent[i - 2],
                    '-21:words=' + '|'.join(sent[i - 2: i]),
                    '-210:words=' + '|'.join(sent[i - 2: i + 1]),
                ])

            if i < length - 1:
                features.extend([
                    '+1:word=' + sent[i + 1],
                    '+01:words=' + '|'.join(sent[i: i + 2]),
                ])
            else:
                features.append('EOS')

            if i < length - 2:
                features.extend([
                    '+2:word=' + sent[i + 2],
                    '+12:words=' + '|'.join(sent[i + 1: i + 3]),
                    '+012:chars=' + '|'.join(sent[i: i + 3]),
                ])

            if 0 < i < length - 1:
                features.extend([
                    '-11:words=' + sent[i - 1] + '|' + sent[i + 1],
                    '-101:words=' + '|'.join(sent[i - 1: i + 2]),
                ])

            feat_list.append(features)

        return feat_list

    @staticmethod
    def get_word_pattern(word):
        pattern = ''
        char = ''
        i = -1
        for ch in word:
            if ch != char:
                i += 1
            pattern += ascii_uppercase[i]
            char = ch
        return pattern

    def sent2tags(self, sent):
        pass

    def train(self, train_x, train_y, out_model):
        trainer = pycrfsuite.Trainer(verbose=False)
        for x, y in zip(train_x, train_y):
            if x and y:
                trainer.append(x, y)

        trainer.set_params({
            'c1': 1.0,                            # coefficient for L1 penalty
            'c2': 1e-3,                           # coefficient for L2 penalty
            'max_iterations': 50,                 # stop earlier
            'feature.possible_transitions': True  # include transitions that are possible, but not observed
        })

        trainer.train(out_model)
        print(trainer.logparser.last_iteration)

    def build_data(self, data_file):
        X = []
        Y = []

        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    x, y = line.split('\t')
                    feat_list = self.sent2features(x.split())
                    tag_list = y.split()
                    X.append(feat_list)
                    Y.append(tag_list)

        return X, Y

    def split_data(self, X, Y):
        random.seed(42)
        rd_num = random.random()

        def _rd():
            return rd_num

        random.shuffle(X, _rd)
        random.shuffle(Y, _rd)

        ratio = round(len(X) * 0.9)
        return X[:ratio], Y[:ratio], X[ratio:], Y[ratio:]

    def eval(self, test_x, test_y, crf_model):
        tagger = pycrfsuite.Tagger()
        tagger.open(crf_model)

        y_pred = []
        for feat_list in test_x:
            preds = tagger.tag(feat_list)
            y_pred.append(preds)

        lb = LabelBinarizer()
        y_true_all = lb.fit_transform(list(chain.from_iterable(test_y)))
        y_pred_all = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = sorted(set(lb.classes_))
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        print(classification_report(
            y_true_all,
            y_pred_all,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
            digits=5
        ))

    def postag(self, sent):
        feat_list = self.sent2features(sent)
        tags = self.tagger.tag(feat_list)
        return tags


