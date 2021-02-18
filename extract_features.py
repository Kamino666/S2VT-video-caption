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


def extract_feats(frame_path, feats_path, interval=10):
    """
    从frame中提取feature
    :param frame_path: frame路径
    :param feats_path: 保存feat的路径
    :param interval: 视频取样的间隔
    :return:
    """
    # load model
    model = pretrainedmodels.resnet152(pretrained='imagenet')
    model.my_avgpool = nn.AdaptiveAvgPool2d([1, 1])
    model = model.to(device)
    load_image_fn = utils.LoadTransformImage(model)

    # load data
    videos = list(plb.Path(frame_path).glob('*'))
    # for index, video in enumerate(tqdm(videos, desc='extracting feats')):
    print("extracting feats of {} videos".format(len(videos)))
    for index, video in enumerate(videos):
        img_list = sorted(video.glob('*.jpg'))
        # 用np的linspace来获取样本下标
        samples_ix = np.arange(0, len(img_list), interval)
        print("samples index of {} is {}".format(index, str(samples_ix)))
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
        # with open(os.path.join(feats_path, video.name + ".npy"), 'w') as f:
        #     pass
        np.save(os.path.join(feats_path, video.name + ".npy"), feats)
    print("extract feats successfully")


def extract(video_path, frame_path, feats_path):
    """
    :param feats_path:
    :param frame_path:
    :param video_path: (str)
    :return:
    """
    # get paths and get frames
    video_path = plb.Path(video_path)
    assert video_path.is_dir()
    # video_to_id = {video.stem: i for i, video in enumerate(video_path.iterdir())}
    # id_to_video = {v: k for k, v in video_to_id.items()}
    # for video in tqdm(video_path.iterdir(), desc='extracting frames'):
    #     extract_frames(str(video), os.path.join(frame_path, str(video.stem)))

    # get features
    extract_feats(frame_path, feats_path)

    # # save video to id
    # with open('./data/video2id.json', 'w+', encoding='utf-8') as f:
    #     json.dump(video_to_id, f)


if __name__ == '__main__':
    print("Extract Features of MSVD with resnet152, GPU: {}".format(str(device)))
    extract(
        video_path=r"./data/YouTubeClips",
        frame_path=r'./data/frames',
        feats_path=r'./data/feats'
    )
