from math import log10

from jiayan.globals import re_zh_include, re_whitespace, stopchars

"""
Use HMM to consider word detection as a char sequence tagging problem.

With a word dict and a char sequence, there could be lots of tokenizing solutions, and the best one will have
the biggest multiplication probability of words:
(see Max Probability Tokenizing: [https://blog.csdn.net/u010189459/article/details/37956689])
p(S) = p(w1) * p(w2) * p(w3)...p(wn)

However, without a word dict we don't know how to tokenize the sentence by word. But here we can use
language model to compute a possible word probability first:
p(w) = p(c1, c2, c3, c4) = p(c1) * p(c2|c1) * p(c3|c1, c2) * p(c4|c1, c2, c3)

Here the word "w" is a 4-char word, with c1, c2, c3 and c4, and the probabilities of each char occurring in relative
position could be computed with N-grams model.

So assume the longest word we want is 4-char word, then in a sentence with length L (L char sequence), each char 
could be in 4 possible positions of one word, and each associates with its probability of being at that position
(k indicates the kth char in the sequence)

1. the beginning of the word (b): p(ck)
2. the second char of the word (c): p(ck|ck-1)
3. the third char of the word (d): p(ck|ck-2, ck-1)
4. the fourth char of the word (e): p(ck|ck-3, ck-2, ck-1)

So, a char sequence could be tagged in a char level with labels {b, c, d, e} first, and be chunked based on the
tags. Now we can see the word level problem is broken down to char level problem with hidden states, so this is the 
decoding problem of HMM, we can use viterbi algorithm to get the best tag/state sequence for the char/observation
sequence.

For viterbi, we need (a)initial starting probabilities of each state, (b)transition probabilities between states, and
(c)emission probabilities of states emitting different observations. Let's draw a table to see what they should be in
this problem:

----------------------------------------------------
   start      ->    b           b           b
                    c           c           c
                    d           d           d
                    e           e           e

char sequence:    char1       char2        char3 ...
-----------------------------------------------------

So for each char in the sequence, there are 4 possible states.
For (a), only "b" can start a sequence, so p(b|<S>) = 1, and p(c|<S>) = p(d|<S>) = p(e|<S>) = 0
For (b), consider the longest word: "bcde", we can see the state transitions are limited in:
    i.   b -> b, b -> c: the beginning of a word either goes to a new word beginning, or the 2nd char;
    ii.  c -> b, c -> d: the 2nd char either goes to a new word beginning, or the 3rd char;
    iii. d -> b, d -> e: the 3rd char either goes to a new word beginning, or the 4th char;
    iv.  e -> b, e -> e: the 4th char either goes to a new word beginning, or the 5th char ...
For (c), the emission probability of one char at a certain state could be computed with N-grams model, e.g.,
    p(ck|d) = p(ck|ck-1, ck-2)

The only parameters that we cannot compute here are transition probabilities, which we can manually set.

Differences from regular HMM tokenizing:
(a) regular HMM tokenizing uses label set {B, M, E, S} to tag char sequence, which is very vague to indicate
    exact char position within a word, especially "M", thus hard to compute emission probabilities;
(b) regular HMM tokenizing requires large data to compute transition and emission probabilities, but here our
    goal is the opposite, to generate that word corpus;
(c) regular HMM tokenizing computes transition probabilities from data, but here we set them manually;
(d) regular HMM tokenizing computes emission probabilities from data, but here we use char level N-grams 
    language model.

Disadvantages:
(a) slow: read the sentence data to build ngrams from min word length to max word length, and read again to tokenize
          the whole data, and by this to build word corpus; viterbi on each sentence in data
(b) bad at long word: need to fine tune transition probabilities to control the word lengths, which is time consuming,
          and the detected long words are not as good as short words.
(c) fake word frequency: since word corpus is built by tokenizing, which can lead to inaccurate sentence splits, the
          word count doesn't reflect true frequency, e.g., "天下" in "于天下". So we use its true frequency count in 
          the ngrams dict when filtering.
"""


class CharHMMTokenizer:

    def __init__(self, lm):
        self.lm = lm
        self.inits = {'b': 0.0, 'c': -3.14e100, 'd': -3.14e100, 'e': -3.14e100}

        # the transition probabilities are manually fine tuned;
        # in principle, we would like the word length the shorter the better;
        # low to-b and high to-next-char-in-word transition probs lead to long words;
        # high to-b and low to-next-char-in-word transition probs lead to short words.
        trans = {'bb': 0.85, 'bc': 0.15,
                 'cb': 0.9925, 'cd': 0.0075,
                 'db': 0.999, 'de': 0.001,
                 'eb': 0.9999, 'ee': 0.0001}
        # trans = {'bb': 0.8, 'bc': 0.2,
        #          'cb': 0.9925, 'cd': 0.0075,
        #          'db': 0.999, 'de': 0.001,
        #          'eb': 0.9999, 'ee': 0.0001}

        # convert the decimal probabilities to logs to avoid overflow
        self.trans = {states: log10(trans_prob) for states, trans_prob in trans.items()}

    def tokenize(self, text: str):
        """ Gets the tags of given sentence, and tokenizes sentence based on the tag sequence.
        """
        # split chars into char chunks by zh chars
        for chunk in re_zh_include.split(re_whitespace.sub('', text)):
            # if zh chars, tokenize them
            if re_zh_include.match(chunk):
                tags = self.viterbi(chunk)

                word = chunk[0]
                for i in range(1, len(chunk)):
                    if tags[i] == 'b':
                        if not self.valid_word(word):
                            for char in word:
                                yield char
                        else:
                            yield word
                        word = chunk[i]
                    else:
                        word += chunk[i]
                if word:
                    if not self.valid_word(word):
                        for char in word:
                            yield char
                    else:
                        yield word

            # if not zh chars, we assume they are all punctuations, split them
            else:
                for char in chunk:
                    yield char

    def viterbi(self, sent):
        """ Chooses the most likely char tag sequence of given char sentence.
        """
        emits = self.get_emission_probs(sent)

        # record the best path for each state for each char, {path1: path_prob, path2: path_prob, ...};
        # paths grow at each decoding step, eventually contains the best paths for each state of last char;
        # we assume the initial state probs = 1st char's emission probs
        paths = {state: prob + self.inits[state] for state, prob in emits[0].items()}

        # for each char
        for i in range(1, len(sent)):
            # print(paths)

            # record best paths and their probs to all states of current char
            cur_char_paths = {}

            # for each state of current char
            for state, emit_prob in emits[i].items():

                # record all possible paths and their probs to current state
                cur_state_paths = {}

                # for each state of previous char
                for path, path_prob in paths.items():
                    trans_states = path[-1] + state

                    # compute the path prob from a previous state to current state
                    if trans_states in self.trans:
                        cur_state_paths[path + state] = path_prob + emit_prob + self.trans[trans_states]

                # choose the best path from all previous paths to current state
                best_path = sorted(cur_state_paths, key=lambda x: cur_state_paths[x])[-1]

                # for current state of current char, we found its best path
                cur_char_paths[best_path] = cur_state_paths[best_path]

            # the paths grow by one char/state
            paths = cur_char_paths

        return sorted(paths, key=lambda x: paths[x])[-1]

    def get_emission_probs(self, sent):
        """ Computes emission probability of each state emitting relative char in the given char sequence. """
        return [

            {'b': self.seg_prob(sent[i]),
             'c': self.seg_prob(sent[i - 1:i + 1]),
             'd': self.seg_prob(sent[i - 2:i + 1]),
             'e': self.seg_prob(sent[i - 3:i + 1])
             }

            for i in range(len(sent))
        ]

    def seg_prob(self, seg):
        """ Computes the segment probability based on ngrams model.
            If given an empty segment, it means it's impossible for current char to be at current position of a word,
            thus return default low log prob -100.
        """
        return (self.lm.score(' '.join(seg), bos=False, eos=False) -
                self.lm.score(' '.join(seg[:-1]), bos=False, eos=False)) \
            or -100.0
    
    def valid_word(self, word):
        """ Checks if a word contains stopchars, if yes, it's not a valid word. """
        for char in word:
            if char in stopchars:
                return False
        return True


