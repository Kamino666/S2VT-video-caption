import pandas as pd
import numpy as np
import re
import json
from collections import Counter
from tqdm import tqdm


def build_vocab(all_words, min_feq=3):
    # use collections.Counter() to build vocab
    all_words = all_words.most_common()
    word2ix = {'<sos>': 0, '<eos>': 1, '<unk>': 2}
    for ix, (word, feq) in enumerate(tqdm(all_words, desc='building vocab'), start=3):
        if feq < min_feq:
            break
        word2ix[word] = ix
    ix2word = {v: k for k, v in word2ix.items()}

    # output info
    print('number of words in vocab: {}'.format(len(word2ix)))
    print('number of <unk> in vocab: {}'.format(len(all_words) - len(word2ix)))

    return word2ix, ix2word


def parse_csv(csv_file, v2id_file, captions_file):
    """
    解析MSVD数据结构的csv文件
    :param csv_file: path of MSVD videos
    :param v2id_file: video to file的对应json文件
    :param captions_file: 生成的caption保存路径
    :return: None
    """
    # read csv
    file = pd.read_csv(csv_file, encoding='utf-8')
    data = pd.DataFrame(file)
    data = data.dropna(axis=0)
    eng_data = data[data['Language'] == 'English']
    print('There are totally {} english descriptions'.format(len(eng_data)))

    # get valid captions and its video ids
    captions = []
    counter = Counter()
    ids = []
    f = open(v2id_file, encoding='utf-8')
    v2id = json.load(f)
    """遍历所有的英文描述，找到feature包含的video对应的caption，存储。"""
    for _, name, start, end, sentence in tqdm(eng_data[['VideoID', 'Start', 'End', 'Description']].itertuples(),
                                              desc='reading captions'):
        # get id
        file_name = name + '_' + str(start) + '_' + str(end) + '.avi'
        if file_name not in v2id:
            continue
        ids.append(v2id[file_name])
        # process caption
        sentence = sentence.lower()
        sentence = re.sub(r'[.!,;?:]', ' ', sentence).split()
        counter.update(sentence)  # put words into a set
        captions.append(['<sos>'] + sentence + ['<eos>'])
    f.close()

    # build vocab
    word2ix, ix2word = build_vocab(counter)

    # turn words into index (2 is <unk>)
    captions = [[word2ix.get(w, 2) for w in caption]
                for caption in tqdm(captions, desc='turing words into index')]

    # get dict of id: [captions]
    caption_dict = {}
    for id, cap in zip(ids, captions):
        if id not in caption_dict.keys():
            caption_dict[id] = []
        caption_dict[id].append(cap)

    # save files
    with open(captions_file, 'w+', encoding='utf-8') as f:
        json.dump(
            {'word2ix': word2ix,
             'ix2word': ix2word,
             'captions': caption_dict}, f
        )


def human_test(test_num, v2id_file, captions_file):
    """
    随机抽取test_num个视频以及对应的一个Caption，手动查看是否对应成功
    :param test_num:
    :param v2id_file:
    :param captions_file:
    :return:
    """
    import random
    f1 = open(v2id_file, encoding='utf-8')
    f2 = open(captions_file, encoding='utf-8')
    v2id = json.load(f1)
    data = json.load(f2)
    for i in range(test_num):
        sample = random.choice(list(v2id.keys()))
        print("choose sample: {}: {}".format(sample, v2id[sample]))
        caption = random.choice(data['captions'][str(v2id[sample])])
        caption = [data['ix2word'][str(w)] for w in caption]
        print(' '.join(caption))

    f1.close()
    f2.close()


if __name__ == '__main__':
    human_test(5, v2id_file=r'./data/video2id.json', captions_file=r'./data/captions.json')
    # parse_csv(
    #     csv_file=r'./data/video_corpus.csv',
    #     v2id_file=r'./data/video2id.json',
    #     captions_file=r'./data/captions.json'
    # )
