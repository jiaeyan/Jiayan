import random



def test():
    global total
    for i in range(10):
        total += 1
        yield i


if __name__ == '__main__':
    # def rd():
    #     return 0.3
    # # random.seed(41)
    # a = [1, 2, 3, 7, 9]
    # b = [4, 5, 6, 8, 0]
    # random.shuffle(a, rd)
    # random.shuffle(b, rd)
    # print(a)
    # print(b)
    # a = [1, 4, 2, 6, 9]
    # b = sorted(a)
    # print(b)
    # from pyltp import Segmentor
    #
    # segmentor = Segmentor()
    # segmentor.load("/Users/jiaeyan/Desktop/ltp_data_v3.4.0/cws.model")
    # words = segmentor.segment("吾若受鱼，是反晏子之义，而顺谄谀之欲，固辞鱼不受")
    # print(list(words))

    from pyltp import Postagger

    words = ['能', '说', '诸', '心', '，', '能', '研', '诸侯', '之', '虑', '，', '定', '天下', '之', '吉凶', '。']
    postagger = Postagger()  # 初始化实例
    postagger.load('/Users/jiaeyan/Desktop/ltp_data_v3.4.0/pos.model')  # 加载模型
    postags = postagger.postag(words)  # 词性标注
    print(list(postags))

    # segmentor.release()
    postagger.release()  # 释放模型
