import os
import marshal
from math import log

from jiayan.globals import re_zh_include

"""
References:
[https://github.com/fxsjy/jieba]
[https://blog.csdn.net/daniel_ustc/article/details/48223135]
"""

dir_path = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

dict_path = os.path.join(root, 'data/dict.txt')
cache_path = os.path.join(dir_path, 'tokenizer.cache')


class WordNgramTokenizer:

    def __init__(self, dict_f=None):
        if not dict_f:
            dict_f = dict_path
        self.cache = cache_path
        self.PREFIX, self.total = self.check_cache(dict_f)

    def check_cache(self, dict_f):
        """ Loads frequency dict and total word counts from cache.
        """
        if os.path.isfile(self.cache):
            with open(self.cache, 'rb') as cf:
                return marshal.load(cf)
        else:
            # if no cache, generate freq dict and dump the cache
            PREFIX, total = self.gen_prefix_dict(dict_f)
            with open(self.cache, 'wb', encoding='utf-8') as temp_cache_file:
                marshal.dump((PREFIX, total), temp_cache_file)
            return PREFIX, total

    def clear_cache(self):
        if os.path.isfile(self.cache):
            os.remove(self.cache)

    @staticmethod
    def gen_prefix_dict(dict_f):
        """ Reads a dict file and generates the prefix dictionary with total word counts.
        """
        word_counts = {}
        with open(dict_f, 'rb') as f:
            for line in f:
                line = line.strip().decode('utf-8')
                word, freq = line.split(',')
                word_counts[word] = int(freq)

                # enumerate all prefixes of a word to enrich the vocab
                for i in range(len(word)):
                    prefix = word[:i + 1]
                    if prefix not in word_counts:
                        word_counts[prefix] = 0

        return word_counts, sum(word_counts.values())

    def tokenize(self, text):
        # split zh chars and non-zh chars into chunks
        chunks = re_zh_include.split(text)

        for chk in chunks:
            if chk:
                # if the chunk is zh, tokenize it
                if re_zh_include.match(chk):
                    for word in self.cut_DAG(chk):
                        yield word
                # if the chunk is not zh, treat it as a single word
                else:
                    yield chk

    def cut_DAG(self, sentence):
        """ Cuts the DAG according to max route probabilities.
        """
        DAG = self.gen_DAG(sentence)
        route = {}
        self.calculate_route_prob(sentence, DAG, route)

        start = 0
        N = len(sentence)

        while start < N:
            end = route[start][1]
            word = sentence[start:end + 1]
            yield word
            start = end + 1

    def gen_DAG(self, sentence):
        """ Generates DAG based on given sentence and prefix dict.
        """
        DAG = {}
        N = len(sentence)

        for start in range(N):
            ends = []
            end = start
            prefix = sentence[start]
            while end < N and prefix in self.PREFIX:
                if self.PREFIX[prefix]:
                    ends.append(end)
                end += 1

                # extend prefix
                prefix = sentence[start:end + 1]

            # if no words formed starting from current char, OOV, it ends with itself
            if not ends:
                ends.append(start)

            DAG[start] = ends

        return DAG

    def calculate_route_prob(self, sentence, DAG, route):
        """ Uses dynamic programming to compute the tokenizing solution with highest probability.
        """
        N = len(sentence)

        # each position in the route will be stored as "position: (prob, end)", where the value
        # tuple contains the highest path prob to current position, and the most recent word
        # ending position from current route position;
        # in other words, sentence[position: end + 1] forms the word and together with which
        # the rest of the path that makes the tokenizing solution with highest probability
        route[N] = (0, 0)
        log_total = log(self.total)

        # compute from backwards to forwards, because ...
        for i in range(N - 1, -1, -1):

            # for each word start position, lists all its possible word ending positions,
            # compute their word probabilities, and add relative rest path probabilities,
            # then choose the end position that makes the whole path probability highest

            # the value got from PREFIX dict could be either None or 0, we assume each word
            # appears at least once, like add-1 laplace smoothing
            route[i] = max((log(self.PREFIX.get(sentence[i:end + 1]) or 1) - log_total
                            + route[end + 1][0], end) for end in DAG[i])


if __name__ == '__main__':
    print(dir_path)
    print(root)
    print(dict_path)
    print(cache_path)
    # s = '日月出矣而爝火不息'
    # t = WordNgramTokenizer('dict.txt')
    # dag = t.get_DAG(s)
    #
    # # 打印s的前缀字典
    # print(u'\"%s\"的前缀字典:' % s)
    # for pos in range(len(s)):
    #     print(s[:pos+1], t.gen_word_freq(s[:pos+1]))
    #
    # print(u'\"%s\"的DAG:' % s)
    # for d in dag:
    #     print(d, ':', dag[d])
    # route = {}
    # t.calc(s, dag, route)
    # print('route:')
    # print(route)

    # print(list(t.tokenize('朝菌不知晦朔')))
    # print(list(t.tokenize('日月出矣而爝火不息')))
    # print(list(t.tokenize('背负青天而莫之夭阏者')))
    # print(list(t.tokenize('抟扶摇羊角而上者九万里')))
    # print(list(t.tokenize('翱翔蓬蒿之间')))
    # print(list(t.tokenize('天下从事者不可以无法仪')))
    # print(list(t.tokenize('天下之为父母者众')))
    # print(list(t.tokenize('何以知尚贤之为政本也')))
    # print(list(t.tokenize('以为锦绣文采靡曼之衣')))
    # print(list(t.tokenize('天下大乱，贤圣不明，道德不一，天下多得一察焉以自好，譬如耳目，皆有所明，不能相通，犹百家众技也，皆有所长，时有所用，虽然不该不遍一之士也，判天地之美，析万物之理，察古人之全寡，能备于天地之美，称神之容，是故内圣外王之道，暗而不明，郁而不发，天下之人各为其所欲焉，以自为方，悲夫，百家往而不反，必不合矣，后世之学者不幸不见天地之纯，古之大体，道术将为天下裂')))
    # print(list(t.tokenize('有，足游浮云，背凌苍天，尾偃天间，跃啄北海，颈尾咳于天地乎，然而漻漻不知六翮之所在。')))

