import pandas as pd
import numpy as np
import re
import json
from collections import Counter
import random
from tqdm import tqdm


def build_vocab(all_words, min_feq=3):
    # use collections.Counter() to build vocab
    all_words = all_words.most_common()
    word2ix = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    for ix, (word, feq) in enumerate(tqdm(all_words, desc='building vocab'), start=4):
        if feq < min_feq:
            break
        word2ix[word] = ix
    ix2word = {v: k for k, v in word2ix.items()}

    # output info
    print('number of words in vocab: {}'.format(len(word2ix)))
    print('number of <unk> in vocab: {}'.format(len(all_words) - len(word2ix)))

    return word2ix, ix2word


def parse_csv(csv_file, captions_file):
    """
    解析MSVD数据结构的csv文件
    :param csv_file: path of MSVD videos
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
    filenames = []
    """遍历所有的英文描述，找到feature包含的video对应的caption，存储。"""
    for _, name, start, end, sentence in tqdm(eng_data[['VideoID', 'Start', 'End', 'Description']].itertuples(),
                                              desc='reading captions'):
        # get id
        file_name = name + '_' + str(start) + '_' + str(end)  # + '.avi'
        # if file_name not in v2id:
        #     continue
        filenames.append(file_name)
        # process caption
        sentence = sentence.lower()
        sentence = re.sub(r'[.!,;?:]', ' ', sentence).split()
        counter.update(sentence)  # put words into a set
        captions.append(['<sos>'] + sentence + ['<eos>'])


    # build vocab
    word2ix, ix2word = build_vocab(counter)

    # turn words into index (3 is <unk>)
    captions = [[word2ix.get(w, 3) for w in caption]
                for caption in tqdm(captions, desc='turing words into index')]

    # build dict   filename: [captions]
    caption_dict = {}
    for name, cap in zip(filenames, captions):
        if name not in caption_dict.keys():
            caption_dict[name] = []
        caption_dict[name].append(cap)

    # 划分数据集
    data_split = [1400, 450, -1]  # train test valid
    vid_names = list(caption_dict.keys())
    random.shuffle(vid_names)
    train_split = vid_names[:data_split[0]]
    test_split = vid_names[data_split[0]:data_split[0] + data_split[1]]
    valid_split = vid_names[data_split[0] + data_split[1]:]

    print("train:{} test:{} valid:{}".format(len(train_split), len(test_split), len(valid_split)))

    # save files
    with open(captions_file, 'w+', encoding='utf-8') as f:
        json.dump(
            {'word2ix': word2ix,
             'ix2word': ix2word,
             'captions': caption_dict,
             'splits': {'train': train_split, 'test': test_split, 'valid': valid_split}}, f
        )


def human_test(test_num, captions_file):
    """
    随机抽取test_num个Caption，手动查看是否对应成功
    :param test_num:
    :param captions_file:
    :return:
    """
    import random
    with open(captions_file, encoding='utf-8') as f:
        data = json.load(f)
    for i in range(test_num):
        sample_video = random.choice(list(data['captions'].keys()))
        sample_cap = random.choice(data['captions'][sample_video])
        sample_cap = [data['ix2word'][str(w)] for w in sample_cap]
        print("[{}]: ".format(sample_video), ' '.join(sample_cap))


if __name__ == '__main__':
    parse_csv(
        csv_file=r'./data/video_corpus.csv',
        captions_file=r'./data/captions.json'
    )
    # human_test(5, captions_file=r'./data/captions.json')
