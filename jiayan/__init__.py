import kenlm

from jiayan.lexicon.pmi_entropy_constructor import PMIEntropyLexiconConstructor
from jiayan.tokenizer.hmm_tokenizer import CharHMMTokenizer
from jiayan.tokenizer.ngram_tokenizer import WordNgramTokenizer
from jiayan.sentencizer.crf_sentencizer import CRFSentencizer
from jiayan.sentencizer.crf_punctuator import CRFPunctuator
from jiayan.postagger.crf_pos_tagger import CRFPOSTagger


def load_lm(lm):
    return kenlm.LanguageModel(lm)

