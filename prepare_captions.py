import pandas as pd
import numpy as np
import re
import json
from collections import Counter
import random
from tqdm import tqdm


def build_vocab(all_words, min_feq=1):
    # use collections.Counter() to build vocab
    all_words = all_words.most_common()
    word2ix = {'<pad>': 0, '<unk>': 1}
    for ix, (word, feq) in enumerate(tqdm(all_words, desc='building vocab'), start=2):
        if feq < min_feq:
            continue
        word2ix[word] = ix
    ix2word = {v: k for k, v in word2ix.items()}

    # output info
    print('number of words in vocab: {}'.format(len(word2ix)))
    print('number of <unk> in vocab: {}'.format(len(all_words) - len(word2ix)))

    return word2ix, ix2word


def parse_csv(csv_file, captions_file, gts_file, clean_only=False):
    """
    parse the .csv file in MSVD dataset
    :param gts_file: save path
    :param clean_only: only choose clean data
    :param csv_file: path of MSVD videos
    :param captions_file: save path
    :return: None
    """
    # read csv
    file = pd.read_csv(csv_file, encoding='utf-8')
    data = pd.DataFrame(file)
    data = data.dropna(axis=0)
    eng_data = data[data['Language'] == 'English']
    if clean_only is True:
        eng_data = eng_data[eng_data['Source'] == 'clean']
    print('There are totally {} english descriptions'.format(len(eng_data)))

    # get valid captions and its video ids
    captions = []
    counter = Counter()
    filenames = []
    gts = {}  # for eval.py
    max_cap_ids = {}  # for eval.py
    for _, name, start, end, sentence in tqdm(eng_data[['VideoID', 'Start', 'End', 'Description']].itertuples(),
                                              desc='reading captions'):
        # get id
        file_name = name + '_' + str(start) + '_' + str(end)  # + '.avi'
        filenames.append(file_name)
        # process caption
        tokenized = sentence.lower()
        tokenized = re.sub(r'[~\\/().!,;?:]', ' ', tokenized).split()
        tokenized = ['<sos>'] + tokenized + ['<eos>']
        counter.update(tokenized)  # put words into a set
        captions.append(tokenized)
        # gts
        if file_name in gts:
            max_cap_ids[file_name] += 1
            gts[file_name].append({
                u'image_id': file_name,
                u'cap_id': max_cap_ids[file_name],
                u'caption': sentence,
                u'tokenized': tokenized
            })
        else:
            max_cap_ids[file_name] = 0
            gts[file_name] = [{
                u'image_id': file_name,
                u'cap_id': 0,
                u'caption': sentence,
                u'tokenized': tokenized
            }]

    # build vocab
    word2ix, ix2word = build_vocab(counter)

    # turn words into index (1 is <unk>)
    captions = [[word2ix.get(w, word2ix['<unk>']) for w in caption]
                for caption in tqdm(captions, desc='turing words into index')]

    # build dict   filename: [captions]
    caption_dict = {}
    for name, cap in zip(filenames, captions):
        if name not in caption_dict.keys():
            caption_dict[name] = []
        caption_dict[name].append(cap)

    # split dataset
    data_split = [1400, 450, -1]  # train valid test
    vid_names = list(caption_dict.keys())
    random.shuffle(vid_names)
    train_split = vid_names[:data_split[0]]
    valid_split = vid_names[data_split[0]:data_split[0] + data_split[1]]
    test_split = vid_names[data_split[0] + data_split[1]:]

    print("train:{} valid:{} test:{}".format(len(train_split), len(valid_split), len(test_split)))

    # save files
    with open(captions_file, 'w+', encoding='utf-8') as f:
        json.dump(
            {'word2ix': word2ix,
             'ix2word': ix2word,
             'captions': caption_dict,
             'splits': {'train': train_split, 'valid': valid_split, 'test': test_split}}, f
        )
    with open(gts_file, 'w+', encoding='utf-8') as f:
        json.dump({'gts': gts}, f)


def parse_msr_vtt(train_source_file, test_source_file, captions_file, gts_file):
    # read data
    with open(train_source_file, encoding='utf-8') as f:
        data = json.load(f)
        sentences = data["sentences"]
        videos = data["videos"]
    with open(test_source_file, encoding='utf-8') as f:
        videos += json.load(f)["videos"]

    # get valid captions and its video ids
    captions = []
    counter = Counter()
    filenames = []
    gts = {}  # for eval.py
    max_cap_ids = {}  # for eval.py
    for item in sentences:
        file_name = item['video_id']
        filenames.append(file_name)
        # process caption
        sentence = ['<sos>'] + item['caption'] + ['<eos>']
        tokenized = sentence.lower()
        tokenized = re.sub(r'[.!,;?:]', ' ', tokenized).split()
        counter.update(tokenized)  # put words into a set
        captions.append(tokenized)
        # gts
        if file_name in gts:
            max_cap_ids[file_name] += 1
            gts[file_name].append({
                u'image_id': file_name,
                u'cap_id': max_cap_ids[file_name],
                u'caption': sentence,
                u'tokenized': tokenized
            })
        else:
            max_cap_ids[file_name] = 0
            gts[file_name] = [{
                u'image_id': file_name,
                u'cap_id': 0,
                u'caption': sentence,
                u'tokenized': tokenized
            }]

    # build vocab
    word2ix, ix2word = build_vocab(counter)

    # turn words into index (1 is <unk>)
    captions = [[word2ix.get(w, word2ix['<unk>']) for w in caption]
                for caption in tqdm(captions, desc='turing words into index')]

    # build dict   filename: [captions]
    caption_dict = {}
    for name, cap in zip(filenames, captions):
        if name not in caption_dict.keys():
            caption_dict[name] = []
        caption_dict[name].append(cap)

    # split the dataset
    train_split = []
    valid_split = []
    test_split = []
    for video in videos:
        if video['split'] == "train":
            train_split.append(video['video_id'])
        elif video['split'] == "validate":
            valid_split.append(video['video_id'])
        else:
            test_split.append(video['video_id'])
    print("train:{} valid:{} test:{}".format(len(train_split), len(valid_split), len(test_split)))

    # save files
    with open(captions_file, 'w+', encoding='utf-8') as f:
        json.dump(
            {'word2ix': word2ix,
             'ix2word': ix2word,
             'captions': caption_dict,
             'splits': {'train': train_split, 'valid': valid_split, 'test': test_split}}, f
        )
    with open(gts_file, 'w+', encoding='utf-8') as f:
        json.dump({'gts': gts}, f)


if __name__ == '__main__':
    parse_csv(
        csv_file=r'./data/video_corpus.csv',
        captions_file=r'./data/captions.json',
        gts_file=r"./data/gts.json",
        clean_only=True
    )
    # parse_msr_vtt(
    #     train_source_file=r"train_val_videodatainfo.json",
    #     test_source_file=r"test_videodatainfo.json",
    #     gts_file=r"./data/gts.json",
    #     captions_file=r'./data/captions_MSR_VTT.json'
    # )

