from jiayan import PMIEntropyLexiconConstructor
from jiayan import CharHMMTokenizer
from jiayan import WordNgramTokenizer
from jiayan import CRFSentencizer
from jiayan import CRFPunctuator
from jiayan import CRFPOSTagger
from jiayan import load_lm


def construct_lexicon(data_file: str, out_f: str):
    constructor = PMIEntropyLexiconConstructor()
    lexicon = constructor.construct_lexicon(data_file)
    constructor.save(lexicon, out_f)


def hmm_tokenize(lm_path: str, text: str):
    lm = load_lm(lm_path)
    tokenizer = CharHMMTokenizer(lm)
    print(list(tokenizer.tokenize(text)))


def ngram_tokenize(text: str):
    tokenizer = WordNgramTokenizer()
    print(list(tokenizer.tokenize(text)))


def crf_sentencize(lm_path: str, cut_model, text):
    lm = load_lm(lm_path)
    sentencizer = CRFSentencizer(lm)
    sentencizer.load(cut_model)
    print(sentencizer.sentencize(text))


def crf_punctuate(lm_path, cut_model, punc_model, text):
    lm = load_lm(lm_path)
    punctuator = CRFPunctuator(lm, cut_model)
    punctuator.load(punc_model)
    print(punctuator.punctuate(text))


def train_sentencizer(lm_path, data_file, out_model):
    lm = load_lm(lm_path)
    sentencizer = CRFSentencizer(lm)
    print('Building data...')
    X, Y = sentencizer.build_data(data_file)
    train_x, train_y, test_x, test_y = sentencizer.split_data(X, Y)
    X[:] = []
    Y[:] = []
    print('Training...')
    sentencizer.train(train_x, train_y, out_model)
    sentencizer.eval(test_x, test_y, out_model)


def train_punctuator(lm_path, data_file, cut_model, out_model):
    lm = load_lm(lm_path)
    punctuator = CRFPunctuator(lm, cut_model)
    print('Building data...')
    X, Y = punctuator.build_data(data_file)
    train_x, train_y, test_x, test_y = punctuator.split_data(X, Y)
    X[:] = []
    Y[:] = []
    print('Training...')
    punctuator.train(train_x, train_y, out_model)
    punctuator.eval(test_x, test_y, out_model)


def train_postagger(lm_path, data_file, pos_model):
    lm = load_lm(lm_path)
    postagger = CRFPOSTagger(lm)
    print('Building data...')
    X, Y = postagger.build_data(data_file)
    train_x, train_y, test_x, test_y = postagger.split_data(X, Y)
    X[:] = []
    Y[:] = []
    print('Training...')
    postagger.train(train_x, train_y, pos_model)
    postagger.eval(test_x, test_y, pos_model)

if __name__ == '__main__':
    test_f = '天下大乱贤圣不明道德不一天下多得一察焉以自好譬如耳目皆有所明不能相通犹百家众技也皆有所长时有所用虽然不该不遍一之士也判天地之美析万物之理察古人之全寡能备于天地之美称神之容是故内圣外王之道暗而不明郁而不发天下之人各为其所欲焉以自为方悲夫百家往而不反必不合矣后世之学者不幸不见天地之纯古之大体道术将为天下裂'
    test_f1 = '圣人之治民也先治者强先战者胜夫国事务先而一民心专举公而私不从赏告而奸不生明法而治不烦能用四者强不能用四者弱夫国之所以强者政也主之所以尊者权也故明君有权有政乱君亦有权有政积而不同其所以立异也故明君操权而上重一政而国治故法者王之本也刑者爱之自也'
    test_f2 = '公曰善吾不食谄人以言也以鱼五十乘赐弦章章归鱼车塞途抚其御之手曰昔者晏子辞党当作赏以正君故过失不掩之今诸臣谀以干利吾若受鱼是反晏子之义而顺谄谀之欲固辞鱼不受君子曰弦章之廉晏子之遗行也'
    test_f3 = '景公游于菑闻晏子死公乘侈舆服繁驵驱之而因为迟下车而趋知不若车之速则又乘比至于国者四下而趋行哭而往伏尸而号'
    test_f4 = '有足游浮云背凌苍天尾偃天间跃啄北海颈尾咳于天地乎然而漻漻不知六翮之所在'
    test_f5 = '谁知林栖者闻风坐相悦草木有本心何求美人折'
    test_f6 = '能说诸心能研诸侯之虑定天下之吉凶成天下之亹亹者是故变化云为吉事有祥象事知器占事知来天地设位圣人成能人谋鬼谋百姓与能八卦以象告爻彖以情言刚柔杂居而吉凶可见矣'
    test_f7 = '至哉坤元万物资生乃顺承天坤厚载物德合无疆含弘光大品物咸亨牝马地类行地无疆柔顺利贞君子攸行先迷失道后顺得常'
    test_f8 = '天下熙熙一盈一虚一治一乱所以然者何也其君贤不肖不等乎其天时变化自然乎'
    test_f9 = '先生之言悖龙之所以为名者乃以白马之论尔今使龙去之则无以教焉且欲师之者以智与学不如也今使龙去之此先教而后师之也先教而后师之者悖且白马非马乃仲尼之所取龙闻楚王张繁弱之弓载忘归之矢以射蛟兕于云梦之圃而丧其弓左右请求之'
    test_f10 = '伪学伪才揣摩以逢主意从前洋务穆彰阿倾排异己殊堪痛恨若一旦置之重法实有不忍着从宽革职永不叙用于是主战主和之功罪是非千秋论定而枋政之臣欲以掩天下后世之耳目不可得矣'
    test_f11 = '传字世文至圣四十七代孙建炎初随孔端友南渡遂流寓衢州'
    test_f12 = '若乃厯代褒崇之典累朝班赉之恩宠数便蕃固可以枚陈而列数以至验祖壁之遗书访阙里之陈迹荒墟废址沦没于春芜秋草之中者阙有之故老世传之将使闻见之所未尝者如接于耳目之近'
    test_f13 = '颂曰元始二妃帝尧之女嫔列有虞承舜于下以尊事卑终能劳苦瞽叟和宁卒享福祜'
    test_f14 = '弃母姜嫄者邰侯之女也当尧之时行见巨人迹好而履之归而有娠浸以益大心怪恶之卜筮禋祀以求无子终生子'
    test_f15 = '颂曰契母简狄敦仁励翼吞卵产子遂自修饰教以事理推恩有德契为帝辅盖母有力'
    test_f16 = '堂之下则有大冶长老桃花茶巢元脩菜何氏丛橘种秔稌莳枣栗有松期为可斫种麦以为奇事作陂塘植黄桑皆足以供先生之岁用而为雪堂之胜景云耳'
    test_f17 = '占者乡塾里闾亦各有史所以纪善恶而垂劝戒后世惟天于有太史而庶民之有德业者非附贤士大夫为之纪其闻者蔑焉'
    test_f18 = '东家杂记孔子四十七代孙孔传所述杂记曰周灵王二十一年已酉岁即鲁襄公二十二年也当襄公二十二年冬十月庚子日先圣生又曰周敬王四十一年辛酉岁即鲁哀公十六年也当哀公十六年夏四月乙丑日先圣薨先儒以为已丑者误也'
    test_f19 = '周灵王二十一年已酉岁即鲁襄公二十二年也当襄公二十二年冬十月庚子日先圣生是夕有二龙绕室五老降庭五老者五星之精也又颜氏之房闻奏钧天之乐空中有声云天感生圣子故降以和乐笙镛之音'
    test_f20 = '河山大地未尝可以法空也佛必欲空之而屹然沛然卒不能空兵刑灾祸未尝可以度也佛必欲度之而伏尸百万'
    test_f21 = '朱子曰心之虚灵知觉一而已矣而以为有心人道心之异者以其或生于形气之私或原于性命之正而所以为知觉者不同是以或危殆而不安或微妙而难见尔'
    test_f22 = '真西山读书记曰此武王伐纣之事诗意虽主伐纣而言然学者平居讽咏其辞凛然如上帝之实临其上则所以为闲邪存诚之助顾不大哉'
    test_f23 = '述叙既讫乃为主客发其例曰客问主人曰伪经何以名之新学也汉艺文志号为古经五经异义称为古说诸书所述古文尤繁'
    test_f24 = '取胡氏传一句两句为旨而以经事之相类者合以为题传为主经为客有以彼经证此经之题有用彼经而隐此经之题于是此一经者为射覆之书而春秋亡矣'
    test_f25 = '谁非黄帝尧舜之子孙而至于今日其不幸而为臧获为婢妾为舆台皂隶窘穷迫逼无可奈何非其数十代以前即自臧获婢妾舆台皂隶来也一旦奋发有为精勤不倦有及身而富贵者矣及其子孙而富贵者矣'
    test_f26 = '人器有德人和伦常社器有德族谐国安灵器有德则天伦如仪器无德人怨族乱国沸天地失道也'
    test_f27 = '先圣没逮今一千五百余年传世五十或问其姓则内求而不得或审其家则舌举而不下为之后者得无愧乎'
    test_f28 = '高辛父曰蟜极蟜极父曰玄嚣玄嚣父曰黄帝'
    test_f29 = '以为锦绣文采靡曼之衣'
    test_f30 = '通玄理而不通禅必受固执之病通禅理而不通儒多成狂慧之流求其禅儒皆通而又能贯之以道不但今鲜其人即古之紫衣黄冠下除紫阳莲池外恒不多觏'
    tests = [
        test_f, test_f1, test_f2, test_f3, test_f4, test_f5, test_f6, test_f7, test_f8,
        test_f9, test_f10,
        test_f11, test_f12, test_f13, test_f14, test_f15, test_f16, test_f17, test_f18, test_f19, test_f20,
        test_f21, test_f22,
        test_f23, test_f24, test_f25, test_f26, test_f27, test_f28, test_f29, test_f30
    ]





    # train_sentencizer('data/jiayan.klm', '/Users/jiaeyan/Desktop/chn_data/all.txt', 'crf_cut_multi')
    # train_punctuator('data/jiayan.klm', '/Users/jiaeyan/Desktop/chn_data/all.txt', 'crf_cut', 'crf_punc_2')
    train_postagger('data/jiayan.klm', '/Users/jiaeyan/Desktop/chn_data/pos_all.txt', 'pos_model')

    # lm = load_lm('data/jiayan.klm')

    # sentcizer = CRFSentencizer(lm)
    # sentcizer.load("/Users/jiaeyan/Desktop/cut_model")
    # for test in tests:
    #     print(sentcizer.sentencize(test))


    # punctuator = CRFPunctuator(lm, '/Users/jiaeyan/Desktop/cut_model')
    # punctuator.load('/Users/jiaeyan/Desktop/punc_model')
    # for test in tests:
    #     print(punctuator.punctuate(test))

    # tokenizer = CharHMMTokenizer(lm)
    # for test in tests:
    #     print(list(tokenizer.tokenize(test)))





    # test = '天下大乱贤圣不明道德不一天下多得一察焉以自好譬如耳目皆有所明不能相通犹百家众技也皆有所长时有所用虽然不该不遍一之士也' \
    #        '判天地之美析万物之理察古人之全寡能备于天地之美称神之容是故内圣外王之道暗而不明郁而不发天下之人各为其所欲焉以自为方' \
    #        '悲夫百家往而不反必不合矣后世之学者不幸不见天地之纯古之大体道术将为天下裂'
    #
    # lm_path = 'data/jiayan.klm'
    #
    # print('Constructing lexicon...')
    # construct_lexicon('data/庄子.txt', '庄子1.csv')
    #
    # print('\nTokenizing test text with HMM...')
    # hmm_tokenize(lm_path, test)
    #
    # print('\nTokenizing test text with N-grams...')
    # for test in tests:
    #     ngram_tokenize(test)
    #
    # print('\nSentencizing test text with CRF...')
    # crf_sentencize(lm_path, 'crf_cut', test)
    #
    # print('\nPunctuating test text with CRF...')
    # crf_punctuate(lm_path, 'crf_cut', 'crf_punc', test)
