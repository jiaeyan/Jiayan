import random
from itertools import chain

import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


class CRFSentTagger:

    def __init__(self, lm):
        self.lm = lm
        self.tagger = None

        # for feature extraction of punctuator
        self.punc2tag = {
            '。': 'J',
            '！': 'G',
            '？': 'W',
            '，': 'D',
            '、': 'U',
            '：': 'A',
            '；': 'F',
        }

        # for decoding of punctuator
        self.tag2punc = {
            'J': '。',
            'G': '！',
            'W': '？',
            'D': '，',
            'U': '、',
            'A': '：',
            'F': '；',
        }

    def load(self, crf_model):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(crf_model)

    def sent2features(self, sent: str, tags=None):
        pass

    def sent2tags(self, sent: str, punc=''):
        single_tag = 'S'
        end_tag = 'E'

        if punc:
            single_tag = self.punc2tag[punc]
            end_tag = self.punc2tag[punc]

        length = len(sent)
        if length == 1:
            tags = [single_tag]
        elif length == 2:
            tags = ['B', end_tag]
        elif length == 3:
            tags = ['B', 'E2', end_tag]
        elif length == 4:
            tags = ['B', 'E3', 'E2', end_tag]
        elif length == 5:
            tags = ['B', 'M', 'E3', 'E2', end_tag]
        else:
            tags = ['B'] + ['M'] * (length - 4) + ['E3', 'E2', end_tag]

        return tags

    def get_pmi(self, seg):
        pmi = self.lm.score(' '.join(seg), eos=False, bos=False) - \
              (self.lm.score(seg[0], eos=False, bos=False) + self.lm.score(seg[1], eos=False, bos=False))
        if pmi >= 2:
            return '2'
        elif pmi >= 1.5:
            return '1.5'
        elif pmi >= 1:
            return '1'
        elif pmi >= 0.5:
            return '0.5'
        return '0'

    def get_ttest(self, seg):
        former = self.lm.score(' '.join(seg[:2]), eos=False, bos=False) - self.lm.score(seg[0], eos=False, bos=False)
        latter = self.lm.score(' '.join(seg[1:]), eos=False, bos=False) - self.lm.score(seg[1], eos=False, bos=False)
        diff = former - latter
        if diff > 0:
            return 'l'
        elif diff == 0:
            return 'u'
        else:
            return 'r'

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
        pass

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


