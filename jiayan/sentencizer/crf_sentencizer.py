from itertools import chain

from jiayan.globals import re_puncs_exclude
from jiayan.globals import get_char_pos_dict
from jiayan.utils import text_iterator
from jiayan.sentencizer.crf_sent_tagger import CRFSentTagger


class CRFSentencizer(CRFSentTagger):

    def __init__(self, lm):
        super(CRFSentencizer, self).__init__(lm)
        self.char_pos_dict = get_char_pos_dict()

    # def get_char_poss(self, char):
    #     if char in self.char_pos_dict:
    #         # return ''.join(sorted(self.char_pos_dict[char]))
    #         return self.char_pos_dict[char][0]
    #     return 'UNK'

    def sent2features(self, sent: str, tags=None):
        length = len(sent)
        feat_list = []
        for i, char in enumerate(sent):
            features = [
                'bias',
                '0:char=' + char,
                # '0:poss=' + self.get_char_poss(char)
            ]

            if i > 0:
                features.extend([
                    '-1:char=' + sent[i - 1],
                    '-10:chars=' + sent[i - 1: i + 1],
                    '-10:pmi=' + self.get_pmi(sent[i - 1: i + 1]),

                    # '-1:poss=' + self.get_char_poss(sent[i - 1]),
                ])
            else:
                features.append('BOS')

            if i > 1:
                features.extend([
                    # '-2:char=' + sent[i - 2],
                    '-21:chars=' + sent[i - 2: i],
                    '-210:chars=' + sent[i - 2: i + 1],
                ])

            if i < length - 1:
                features.extend([
                    '+1:char=' + sent[i + 1],
                    '+01:chars=' + sent[i: i + 2],
                    '+01:pmi=' + self.get_pmi(sent[i: i + 2]),

                    # '+1:poss=' + self.get_char_poss(sent[i + 1]),
                ])
            else:
                features.append('EOS')

            if i < length - 2:
                features.extend([
                    # '+2:char=' + sent[i + 2],
                    '+12:chars=' + sent[i + 1: i + 3],
                    '+012:chars=' + sent[i: i + 3],
                ])

            if 0 < i < length - 1:
                features.extend([
                    '-11:chars=' + sent[i - 1] + sent[i + 1],
                    '-101:chars=' + sent[i - 1: i + 2],
                    '-101:ttest=' + self.get_ttest(sent[i - 1: i + 2]),
                ])

            feat_list.append(features)

        return feat_list

    def sentencize(self, text):
        feat_list = self.sent2features(text)
        tags = self.tagger.tag(feat_list)

        sents = []
        sent = ''
        for i, tag in enumerate(tags):
            if tag == 'S':
                if sent:
                    sents.append(sent)
                    sent = ''
                sents.append(text[i])
            elif tag == 'B':
                if sent:
                    sents.append(sent)
                sent = text[i]
            elif tag in {'M', 'E3', 'E2', 'E'}:
                sent += text[i]
        if sent:
            sents.append(sent)

        return sents

    def build_data(self, data_file):
        X = []
        Y = []

        for line in text_iterator(data_file, keep_punc=True):
            sents = [sent for sent in re_puncs_exclude.split(line) if sent]
            feat_list = self.sent2features(''.join(sents))
            tag_list = list(chain.from_iterable([self.sent2tags(sent) for sent in sents]))
            X.append(feat_list)
            Y.append(tag_list)

        return X, Y


