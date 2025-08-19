#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn

import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import clip


from PIL import Image

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Shift Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument('-Experiment_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    # parser.add_argument(
    #     '--print-log',
    #     type=str2bool,
    #     default=True,
    #     help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    # parser.add_argument(
    #     '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    
        
    parser.add_argument('--feature_output_dir', default=0)
    
    return parser




# def extract_batch_video_features(video_paths, processor, model):
#     max_frames = 16
#     frame_size = 224
#     all_video_feats = []

#     dummy_feat = np.zeros((1568, 768), dtype=np.float32)  # 你也可以改为 np.random.randn(1568,768)

#     for video_path in video_paths:
#         if video_path is None:
#             print("video_path is None, 使用占位特征。")
#             all_video_feats.append(dummy_feat)
#             continue

#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         while len(frames) < max_frames:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, (frame_size, frame_size))
#             frames.append(frame)

#         cap.release()

#         if len(frames) == 0:
#             print(f"视频为空或无法读取: {video_path}，使用占位特征。")
#             all_video_feats.append(dummy_feat)
#             continue

#         if len(frames) < max_frames:
#             frames += [frames[-1]] * (max_frames - len(frames))

#         inputs = processor(frames, return_tensors="pt")
#         inputs = {k: v.cuda() for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = model(**inputs)
#             token_feats = outputs.last_hidden_state.squeeze(0).cpu().numpy()

#         all_video_feats.append(token_feats)

#     return all_video_feats


# Video MAE
def extract_batch_video_features(video_paths, processor, model):
    max_frames = 16
    frame_size = 224
    all_video_feats = []

    dummy_feat = np.zeros((1568, 768), dtype=np.float32)  # 你也可以改为 np.random.randn(1568,768)

    for video_path in video_paths:
        if video_path is None:
            print("video_path is None, 使用占位特征。")
            all_video_feats.append(dummy_feat)
            continue

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            print(f"视频 {video_path} 没有读取到帧，使用占位特征。")
            all_video_feats.append(dummy_feat)
            continue

        # 均匀采样16帧
        if len(frames) >= max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            sampled_frames = [frames[i] for i in idx]
        else:
            # 不足16帧则重复最后一帧补齐
            sampled_frames = frames + [frames[-1]] * (max_frames - len(frames))

        inputs = processor(sampled_frames, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            video_feat = outputs.last_hidden_state.squeeze(0).cpu().numpy()

        all_video_feats.append(video_feat)

    return all_video_feats


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim, n_heads=8, num_layers=2):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, T, D]
        out = self.transformer(x)
        # 取 [CLS] token 或平均池化作为视频表示
        # return out.mean(dim=1)  # [B, D]
        return out

def extract_clip_transformer_features(video_paths, clip_model, clip_preprocess, temporal_model, device='cuda'):
    max_frames = 16
    frame_size = 224
    all_video_feats = []

    for video_path in video_paths:
        if video_path is None:
            print("video_path is None，使用占位特征。")
            dummy_feat = np.zeros((clip_model.visual.output_dim,), dtype=np.float32)
            all_video_feats.append(dummy_feat)
            continue

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            print(f"视频 {video_path} 无法读取，使用占位特征。")
            dummy_feat = np.zeros((clip_model.visual.output_dim,), dtype=np.float32)
            all_video_feats.append(dummy_feat)
            continue

        # 均匀采样16帧
        if len(frames) >= max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            sampled_frames = [frames[i] for i in idx]
        else:
            sampled_frames = frames + [frames[-1]] * (max_frames - len(frames))

        # 每帧通过 CLIP 提取特征
        clip_feats = []
        for frame in sampled_frames:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            img = clip_preprocess(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = clip_model.encode_image(img).cpu()
            clip_feats.append(img_feat)
        
        clip_feats = torch.stack(clip_feats, dim=1).squeeze(0)  # [T, D]
        clip_feats = clip_feats.unsqueeze(0).to(device)  # [1, T, D]
        clip_feats = clip_feats.float()

        # 通过 temporal transformer
        with torch.no_grad():
            video_feat = temporal_model(clip_feats)  # [1, D]
            video_feat = video_feat.squeeze(0).cpu().numpy()

        all_video_feats.append(video_feat)

    return all_video_feats



class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):

        arg.model_saved_name = "./save_models/"+arg.Experiment_name
        arg.work_dir = "./work_dir/"+arg.Experiment_name
        self.arg = arg
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)

        self.global_step = 0
        self.load_model()
        self.load_data()
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            print(len(Feeder(**self.arg.train_feeder_args)))
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def generate(self,case):
        
        label_spilt_dir='/home/yanghao/data/HKU/hku_skeleton_25/label_splits'
        
        cls_num=60
        us_num=5
        
        output_dir=self.arg.feature_output_dir

        print(output_dir)
        
        s_label=np.load(label_spilt_dir+f'/rs.npy')
        v_label=np.load(label_spilt_dir+f'/rv.npy')
        u_label=np.load(label_spilt_dir+f'/ru.npy')
        
        
        
        self.model.eval()
        loader = self.data_loader['train']
        loss_value = []
        process = tqdm(loader)
        
        temp_dir = '/home/yanghao/temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        if case == 0:
            video_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").cuda()
        elif case == 1:
             # 加载CLIPcd
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

            # 初始化Temporal Transformer，embed_dim根据CLIP输出维度
            temporal_model = TemporalTransformer(embed_dim=clip_model.visual.output_dim).to(device)



        # print(len(loader))
        for batch_idx, (data, video_name, label, index) in enumerate(process):


            if video_name is None:
                continue
            
            temp=[label]
            
            self.global_step += 1

            data = data.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)

            
            # forward
            output, feature = self.model(data)
            temp.append(feature)

            if case == 0:
                token_feats = extract_batch_video_features(video_name, video_processor, video_model)
            elif case == 1:
                token_feats = extract_clip_transformer_features(video_name, clip_model, clip_preprocess, temporal_model)
  
            temp.append(token_feats)

            if isinstance(temp[0], torch.Tensor):
                temp[0] = temp[0].detach().cpu().numpy()
            if isinstance(temp[1], torch.Tensor):
                temp[1] = temp[1].detach().cpu().numpy()
            if isinstance(temp[2], torch.Tensor):
                temp[2] = temp[2].detach().cpu().numpy()

            
            
            
            # np.save(f'{temp_dir}/feature_{batch_idx}.npy', temp)
            
            np.savez(f'{temp_dir}/feature_{batch_idx}.npz', label=temp[0], feature=temp[1], video_feature=temp[2])

            # if batch_idx > 1:
            #     break
        
        feature_files = sorted(os.listdir(temp_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

        features_s = []
        features_u = []
        features_v = []
        labels_s=[]
        labels_u=[]
        labels_v=[]
        video_feats_s = []
        video_feats_u = []
        video_feats_v = []
        for fname in tqdm(feature_files):
            fpath = os.path.join(temp_dir, fname)
            # temp = np.load(fpath,allow_pickle=True)
            # label=temp[0]
            # feat=temp[1]
            
            data = np.load(fpath, allow_pickle=True)
            label = data['label']
            feat = data['feature']
            video_feat = data['video_feature']

            
            for i in range(len(label)):
                # print(f'feat[{i}] shape: {feat[i].shape}')
                if label[i].item() in u_label:
                    features_u.append(feat[i])
                    labels_u.append(label[i])
                    video_feats_u.append(video_feat[i])
                elif label[i].item() in s_label:
                    features_s.append(feat[i])
                    labels_s.append(label[i])
                    video_feats_s.append(video_feat[i])
                if label[i].item() in v_label:
                    features_v.append(feat[i])
                    labels_v.append(label[i])
                    video_feats_v.append(video_feat[i])

        # print(np.array(features_s).shape)
        # print(np.array(labels_s).shape)
        # print(np.array(features_u).shape)
        # print(np.array(labels_u).shape)
        # print(len(features_s))
        # print(len(features_v))

        # all_features = np.concatenate(features,axis=0)  
        # all_labels = np.concatenate(labels, axis=0)
        print(len(features_s))
        print(len(features_u))
        print(len(features_v))
        
        np.save(output_dir+'/train_feature.npy', features_s)
        np.save(output_dir+'/train_labels.npy', labels_s)  
        np.save(output_dir+'/train_video_feature.npy', video_feats_s)
        np.save(output_dir+'/test_feature.npy', features_u)
        np.save(output_dir+'/test_labels.npy', labels_u)
        np.save(output_dir+'/test_video_feature.npy', video_feats_u)
        np.save(output_dir+'/val_feature.npy', features_v)
        np.save(output_dir+'/val_labels.npy', labels_v)     
        np.save(output_dir+'/val_video_feature.npy', video_feats_v) 
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def import_class(name):
    
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod




if __name__ == '__main__':
    parser = get_parser()
    
    

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

   
    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.generate(case=0)
