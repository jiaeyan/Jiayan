from jiayan.globals import re_invalid_chars, re_zh_exclude


def process_line(line: str):
    """ A standard approach to process input line, by
            1. retain and replace valid punctuations;
            2. removing non-zh and invalid punctuation chars;
    """
    line = line.strip().replace(',', '，').replace('.', '。').replace(':', '：').\
        replace('!', '！').replace('?', '？').replace(';', '；')
    line = re_invalid_chars.sub('', line)
    return line


def text_iterator(data_file, keep_punc=False):
    """ A help function to provide clean zh char lines of a given file. """
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = process_line(line)
            if keep_punc:
                if line:
                    yield line
            else:
                for text in re_zh_exclude.findall(line):
                    if text:
                        yield text
