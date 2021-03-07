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
import argparse


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


def extract_feats(frame_path, feats_path, interval, model, video_name):
    """
    extract feature from frames of one video
    :param video_name:
    :param model: name of model
    :param frame_path: path of frames
    :param feats_path: path to store results
    :param interval: (str) The interval when extract frames from videos
    :return: None
    """
    # load model
    C, H, W = 3, 224, 224
    if model == 'resnet152':
        model = pretrainedmodels.resnet152(pretrained='imagenet')
    elif model == 'vgg16':
        model = pretrainedmodels.vgg16(pretrained='imagenet')
    elif model == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(pretrained='imagenet')
    model.last_linear = utils.Identity()
    model = model.to(device)
    model.eval()
    load_image_fn = utils.LoadTransformImage(model)

    # load data
    img_list = sorted(frame_path.glob('*.jpg'))
    # get index
    samples_ix = np.arange(0, len(img_list), interval)
    img_list = [img_list[int(i)] for i in samples_ix]
    # build tensor
    imgs = torch.zeros([len(img_list), C, H, W])
    for i in range(len(img_list)):
        img = load_image_fn(img_list[i])
        imgs[i] = img
    imgs = imgs.to(device)
    with torch.no_grad():
        feats = model(imgs)
    feats = feats.cpu().numpy()
    # save
    np.save(os.path.join(feats_path, video_name + ".npy"), feats)


def fix_frame_extract(frame_path, feats_path, frames_num, model, video_name):
    # load model
    C, H, W = 3, 224, 224
    if model == 'resnet152':
        model = pretrainedmodels.resnet152(pretrained='imagenet')
    elif model == 'vgg16':
        model = pretrainedmodels.vgg16(pretrained='imagenet')
    elif model == 'inception_v4':
        C, H, W = 3, 299, 299
        model = pretrainedmodels.inceptionv4(pretrained='imagenet')
    model.last_linear = utils.Identity()
    model = model.to(device)
    model.eval()
    load_image_fn = utils.LoadTransformImage(model)

    # load data
    img_list = sorted(frame_path.glob('*.jpg'))
    # get index
    samples_ix = np.linspace(0, len(img_list)-1, frames_num).astype(int)
    img_list = [img_list[i] for i in samples_ix]
    # build tensor
    imgs = torch.zeros([len(img_list), C, H, W])
    for i in range(len(img_list)):
        img = load_image_fn(img_list[i])
        imgs[i] = img
    imgs = imgs.to(device)
    with torch.no_grad():
        feats = model(imgs)
    feats = feats.cpu().numpy()
    # save
    np.save(os.path.join(feats_path, video_name + ".npy"), feats)


def extract(video_path, feats_path, model, mode='fix', frames_num=80, interval=10):
    """
    :param interval: only works in mode 'free'
    :param frames_num: only works in mode 'fix'
    :param mode: 'fix': extract a fix number of frames, 'free': extract feat every interval num
    :param model: (str) name of model
    :param feats_path: (str) The path(folder) to store extracted features
    :param video_path: (str) The path of videos
    :return: None
    """
    # check paths
    feats_path = plb.Path(feats_path)
    video_path = plb.Path(video_path)
    temp_path = plb.Path(r"./_frames_out")
    assert video_path.is_dir()
    if feats_path.is_dir():
        shutil.rmtree(feats_path)
    os.mkdir(feats_path)
    if temp_path.is_dir():
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)

    for video in tqdm(list(video_path.iterdir()), desc='Extracting~'):
        # get frames to a temp dir
        extract_frames(str(video), os.path.join(temp_path))
        # get features
        if mode == 'fix':
            fix_frame_extract(temp_path, feats_path, frames_num, model, video.stem)
        elif mode == 'free':
            extract_feats(temp_path, feats_path, interval, model, video.stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, type=str,
                        help="The path of MSVD or MSR-VTT videos")
    parser.add_argument('--mode', required=True, type=str,
                        help="fix or free.\n'fix': extract a fix number of frames. \n'free': extract feat every interval num")
    parser.add_argument('--feat_path', default='./data/feats', type=str,
                        help="The path(folder) to store extracted features")
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'resnet152', 'inception_v4'], type=str,
                        help="The CNN model to extract features. Check Readme file to find support models")
    parser.add_argument('--gpu', default=0, type=int,
                        help="GPU device number(don't support multi GPU)")
    parser.add_argument('--interval', default=10, type=int,
                        help="The interval when extract frames from videos")
    parser.add_argument('--frames_num', default=80, type=int,
                        help="The fix number of frames to be extracted from each video")

    args = parser.parse_args()
    args = vars(args)  # to dict

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Extract Features with {}, device: {}".format(args['model'], str(device)))

    extract(
        video_path=args['video_path'],
        feats_path=args['feat_path'],
        model=args['model'],
        interval=args['interval'],
        frames_num=args['frames_num'],
        mode=args['mode'],
    )

