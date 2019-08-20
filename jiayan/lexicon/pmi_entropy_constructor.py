from math import log2
import time
from jiayan.globals import stopchars
from jiayan.utils import text_iterator

"""
A precise way to discover new words in sentence corpus, consider PMI and entropy.

1. PMI is used to evaluate how tight the two segments of the word;
2. Right and left entropy are used to evaluate how independent the word is in various contexts.

References:
[http://www.matrix67.com/blog/archives/5044]
[https://zhuanlan.zhihu.com/p/25499358]
"""


class Trie:

    class TrieNode:
        def __init__(self):
            self.freq = 0
            self.pmi = 0
            self.r_entropy = 0
            self.l_entropy = 0
            self.children = {}

    def __init__(self):
        self.root = self.TrieNode()

    def add(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.freq += 1

    def find(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return node


class PMIEntropyLexiconConstructor:

    MIN_WORD_LEN = 1
    MAX_WORD_LEN = 4

    # TODO: Different PMI and Entropy thresholds for different lengths
    MIN_WORD_FREQ = 10
    MIN_PMI = 80
    MIN_ENTROPY = 2

    def __init__(self):
        self.trie = Trie()
        self.r_trie = Trie()
        self.total = 0

    def construct_lexicon(self, data_file):
        self.build_trie_trees(data_file)
        self.compute()
        lexicon = self.filter()
        return lexicon

    def build_trie_trees(self, data_file):
        """ Counts frequency of segments of data, also records their left and right char sets.
        """
        max_seg_len = self.MAX_WORD_LEN + 1

        start = time.time()
        for text in text_iterator(data_file):
            length = len(text)
            for i in range(length):
                for j in range(1, min(length - i + 1, max_seg_len + 1)):
                    seg = text[i: i + j]
                    self.trie.add(seg)

                    r_seg = seg[::-1]
                    self.r_trie.add(r_seg)

                    self.total += 1
        end = time.time()

        print('Trie building time:', end - start)

    def compute(self):
        start = time.time()
        node = self.trie.root
        word = ''
        self.compute_help(node, word)
        end = time.time()
        print('Computation time:', end - start)

    def compute_help(self, node, word):
        if node.children:
            for char, child in node.children.items():
                word += char
                if len(word) <= self.MAX_WORD_LEN:
                    self.calculate_pmi(child, word)
                    self.calculate_rl_entropy(child, word)
                    self.compute_help(child, word)
                word = word[:-1]

    def calculate_pmi(self, node, word):
        length = len(word)
        if length == 1:
            node.pmi = self.MIN_PMI
        else:
            constant = node.freq * self.total
            mutuals = (constant / (self.trie.find(word[:i + 1]).freq * self.trie.find(word[i + 1:]).freq)
                       for i in range(length - 1))
            node.pmi = min(mutuals)

    def calculate_rl_entropy(self, node, word):
        # right entropy
        if node.children:
            node.r_entropy = self.calculate_entropy(node)

        # left entropy
        r_word = word[::-1]
        r_node = self.r_trie.find(r_word)
        if r_node.children:
            node.l_entropy = self.calculate_entropy(r_node)

    def calculate_entropy(self, node):
        freqs = [child.freq for child in node.children.values()]
        sum_freqs = sum(freqs)
        entropy = sum([- (x / sum_freqs) * log2(x / sum_freqs) for x in freqs])
        return entropy

    def filter(self):
        """ Filters the PMI and entropy calculation result dict, removes words that do not
            reach the thresholds.
            TODO: test use max of r/l entropy to filter.
        """
        start = time.time()
        node = self.trie.root
        word = ''
        word_dict = {}
        self.filter_help(node, word, word_dict)
        end = time.time()
        print('Word filtering:', end - start)
        return word_dict

    def filter_help(self, node, word, word_dict):
        if node.children:
            for char, child in node.children.items():
                word += char
                if self.valid_word(child, word):
                    word_dict[word] = [child.freq, child.pmi, child.r_entropy, child.l_entropy]
                self.filter_help(child, word, word_dict)
                word = word[:-1]

    def valid_word(self, node, word):
        if self.MIN_WORD_LEN <= len(word) <= self.MAX_WORD_LEN \
                and node.freq >= self.MIN_WORD_FREQ \
                and node.pmi >= self.MIN_PMI \
                and node.r_entropy >= self.MIN_ENTROPY \
                and node.l_entropy >= self.MIN_ENTROPY \
                and not self.has_stopword(word):
            return True
        return False

    def has_stopword(self, word):
        """ Checks if a word contains stopwords, which are not able to construct words.
        """
        if len(word) == 1:
            return False
        for char in word:
            if char in stopchars:
                return True
        return False

    @staticmethod
    def save(lexicon, out_f):
        """ Saves the word detection result in a csv file.
        """
        words = sorted(lexicon, key=lambda x: (len(x), -lexicon[x][0], -lexicon[x][1], -lexicon[x][2], -lexicon[x][3]))
        with open(out_f, 'w') as f:
            f.write('Word,Frequency,PMI,R_Entropy,L_Entropy\n')
            for word in words:
                f.write('{},{},{},{},{}\n'.format(
                    word, lexicon[word][0], lexicon[word][1],
                    lexicon[word][2], lexicon[word][3]))



