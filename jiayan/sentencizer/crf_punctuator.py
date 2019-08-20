from itertools import chain

import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from jiayan.globals import re_puncs_include, re_zh_exclude
from jiayan.utils import text_iterator
from jiayan.sentencizer.crf_sent_tagger import CRFSentTagger
from jiayan.sentencizer.crf_sentencizer import CRFSentencizer


class CRFPunctuator(CRFSentTagger):

    def __init__(self, lm, cut_model):
        super(CRFPunctuator, self).__init__(lm)
        self.sentencizer = CRFSentencizer(lm)
        self.sentencizer.load(cut_model)

    def sent2features(self, sent: str, tags=None):
        length = len(sent)
        feat_list = []
        for i, char in enumerate(sent):
            features = [
                'bias',
                '0:char=' + char,
                '0:tag=' + tags[i],
            ]

            if i > 0:
                features.extend([
                    '-1:char=' + sent[i - 1],
                    '-10:chars=' + sent[i - 1: i + 1],
                    # '-10:pmi=' + self.get_pmi(sent[i - 1: i + 1]),

                    # '-1:tag=' + tags[i - 1],
                    # '-10:tags=' + tags[i - 1: i + 1],
                ])
            else:
                features.append('BOS')

            if i > 1:
                features.extend([
                    '-2:char=' + sent[i - 2],
                    '-21:chars=' + sent[i - 2: i],
                    '-210:chars=' + sent[i - 2: i + 1],

                    # '-21:tags=' + tags[i - 2: i],
                    # '-210:tags=' + tags[i - 2: i + 1],
                ])

            if i > 2:
                features.extend([
                    '-3:char=' + sent[i - 3],
                    '-321:chars=' + sent[i - 3: i],
                    '-3210:chars=' + sent[i - 3: i + 1],
                ])

            if i < length - 1:
                features.extend([
                    '+1:char=' + sent[i + 1],
                    '+01:chars=' + sent[i: i + 2],
                    # '+01:pmi=' + self.get_pmi(sent[i: i + 2]),

                    # '+1:tag=' + tags[i + 1],
                    # '+01:tags=' + tags[i: i + 2],
                ])
            else:
                features.append('EOS')

            if i < length - 2:
                features.extend([
                    '+2:char=' + sent[i + 2],
                    '+12:chars=' + sent[i + 1: i + 3],
                    '+012:chars=' + sent[i: i + 3],

                    # '+12:tags=' + tags[i + 1: i + 3],
                    # '+012:tags=' + tags[i: i + 3],
                ])

            if i < length - 3:
                features.extend([
                    '+3:char=' + sent[i + 3],
                    '+123:chars=' + sent[i + 1: i + 4],
                    '+0123:chars=' + sent[i: i + 4],
                ])

            if 0 < i < length - 1:
                features.extend([
                    '-11:chars=' + sent[i - 1] + sent[i + 1],
                    '-101:chars=' + sent[i - 1: i + 2],
                    '-101:ttest=' + self.get_ttest(sent[i - 1: i + 2]),
                ])

            feat_list.append(features)

        return feat_list

    def punctuate(self, text):
        cut_feat_list = self.sentencizer.sent2features(text)
        cut_tags = self.sentencizer.tagger.tag(cut_feat_list)
        punc_feat_list = self.sent2features(text, cut_tags)
        punc_tags = self.tagger.tag(punc_feat_list)

        sents = []
        sent = ''
        for i, tag in enumerate(punc_tags):
            if tag in self.tag2punc:
                if sent:
                    sents.append(sent)
                    sent = ''
                sents.append(text[i])
                sents.append(self.tag2punc[tag])
            elif tag == 'B':
                if sent:
                    sents.append(sent)
                sent = text[i]
            elif tag in {'M', 'E3', 'E2'}:
                sent += text[i]
        if sent:
            sents.append(sent)

        return ''.join(sents)

    def build_data(self, data_file):
        X = []
        Y = []
        for line in text_iterator(data_file, keep_punc=True):
            texts = [text for text in re_puncs_include.split(line) if text]
            texts = self.process_texts(texts)

            feat_list = []
            punc_tags = []
            for i in range(len(texts) - 1):
                if re_zh_exclude.match(texts[i]) and texts[i + 1] in self.punc2tag:
                    cut_tags = self.sent2tags(texts[i])
                    feat_list.extend(self.sent2features(texts[i], cut_tags))
                    punc_tags.extend(self.sent2tags(texts[i], texts[i + 1]))

            X.append(feat_list)
            Y.append(punc_tags)

        return X, Y

    def process_texts(self, texts):
        while texts and texts[0] in self.punc2tag:
            texts = texts[1:]

        if len(texts) // 2 != 0:
            texts.append('。')

        return texts

    def eval(self, test_x, test_y, crf_model):
        tagger = pycrfsuite.Tagger()
        tagger.open(crf_model)

        pred_y = []
        for feat_list in test_x:
            preds = tagger.tag(feat_list)
            pred_y.append(preds)

        y_trues = [tag for tag in list(chain.from_iterable(test_y)) if tag not in {'B', 'M', 'E3', 'E2'}]
        y_preds = [tag for tag in list(chain.from_iterable(pred_y)) if tag not in {'B', 'M', 'E3', 'E2'}]

        lb = LabelBinarizer()
        y_true_all = lb.fit_transform(y_trues)
        y_pred_all = lb.transform(y_preds)

        tagset = sorted(set(lb.classes_))
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        print(classification_report(
            y_true_all,
            y_pred_all,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
            digits=5
        ))
