import pretrainedmodels
from pretrainedmodels import utils
import torch
import os
import subprocess
import shutil
import pathlib as plb
import numpy as np
from tqdm import tqdm
import json
import h5py
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_frames(video, dst):
    """
    extract all frames of a video to dst
    :param video: path of video
    :param dst: img output file folder
    :return:
    """
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)  # 递归删除文件夹
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]  # %06d 6位数字
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


def extract_feats(frame_path, feats_file, sample_size=40):
    # load model
    model = pretrainedmodels.resnet152(pretrained='imagenet')
    model.my_avgpool = nn.AdaptiveAvgPool2d([1, 1])
    model = model.to(device)
    load_image_fn = utils.LoadTransformImage(model)

    # load data
    videos = list(plb.Path(frame_path).glob('*'))

    # init h5
    feat_h5 = h5py.File(feats_file, 'w')
    ds = feat_h5.create_dataset(name='feats'
                                , shape=[len(videos), sample_size, 2048])

    # load data
    for index, video in enumerate(tqdm(videos, desc='extracting feats')):
        img_list = sorted(video.glob('*.jpg'))
        # 用np的linspace来获取样本下标
        samples_ix = np.round(np.linspace(0, len(img_list) - 1, num=sample_size))
        img_list = [img_list[int(i)] for i in samples_ix]
        # 建立tensor
        imgs = torch.zeros([len(img_list), 3, 224, 224])
        for i in range(len(img_list)):
            img = load_image_fn(img_list[i])
            imgs[i] = img
        imgs = imgs.to(device)
        with torch.no_grad():
            feats = model.features(imgs)
            feats = model.my_avgpool(feats).squeeze(2).squeeze(2)
        feats = feats.cpu().numpy()
        # save to h5
        ds[index, :, :] = feats

    print("There totally {} video features with shape[{},{}]"
          .format(len(videos), sample_size, 2048))
    feat_h5.close()


def extract(video_path, frame_path, feats_file):
    """
    :param feats_file:
    :param frame_path:
    :param video_path: (str)
    :return:
    """
    # get paths and get frames
    video_path = plb.Path(video_path)
    assert video_path.is_dir()
    video_to_id = {video.name: i for i, video in enumerate(video_path.iterdir())}
    id_to_video = {v: k for k, v in video_to_id.items()}
    # for video in tqdm(video_path.iterdir(), desc='extracting frames'):
    #     extract_frames(str(video), os.path.join(frame_path, str(video_to_id[str(video)])))

    # get features
    extract_feats(frame_path, feats_file)

    # save video to id
    with open('./data/video2id.json', 'w+', encoding='utf-8') as f:
        json.dump(video_to_id, f)


if __name__ == '__main__':
    print("Extract Features of MSVD with resnet152, GPU: {}".format(str(device)))
    extract(
        video_path=r"./data/YouTubeClips",
        frame_path=r'./data/frames',
        feats_file=r'./data/feat.h5'
    )
