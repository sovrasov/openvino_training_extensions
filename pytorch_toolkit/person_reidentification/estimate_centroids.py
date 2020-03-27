"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import pdist

import torch.nn as nn
import torch.nn.functional as F

from torchreid.utils import load_pretrained_weights
from data.transforms import build_transforms
from config.default_config import get_default_config, imagedata_kwargs
from models.builder import build_model
from data.datamanager import ImageDataManagerWithTransforms


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = build_model(
            name=cfg.model.name,
            num_classes=1041,  # Does not matter in conversion
            loss=cfg.loss.name,
            pretrained=False,
            use_gpu=True,
            feature_dim=cfg.model.feature_dim,
            fpn_cfg=cfg.model.fpn,
            lct_gate=cfg.model.lct_gate,
            pooling_type=cfg.model.pooling_type,
            input_size=(cfg.data.height, cfg.data.width),
            dropout_cfg=cfg.model.dropout,
            IN_first=cfg.model.IN_first,
            extra_blocks=cfg.model.extra_blocks
        )

    if cfg.model.load_weights:
        load_pretrained_weights(model, cfg.model.load_weights)
        print('Pretrained', cfg.model.load_weights)
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
    model.eval()

    datamanager = ImageDataManagerWithTransforms(**imagedata_kwargs(cfg))
    centroids = {}
    loader = datamanager.trainloader
    loader = datamanager.testloader['msmt17']['gallery']
    j = 0
    for item in tqdm(loader):
        j += 1
        if j > 10000:
            break
        #for idx, item in enumerate(pt_loader):
        #pt_loader = datamanager.trainloader[item] #['gallery']
        embeddings = F.normalize(model(item[0]), dim=1)
        for i in range(item[1].shape[0]):
            label = item[1][i].item()
            if label not in centroids:
                centroids[label] = [torch.zeros((embeddings.shape[1],)).cuda(), 0]
            centroids[label][0] += embeddings[i]
            centroids[label][1] += 1
        # 0 - images 1 - labels 2 - cams 3 - paths

    processed_ids = 0
    max_processed = 200
    selected_centroids = []
    for label in centroids:
        if processed_ids >= max_processed:
            break
        size = centroids[label][1]
        if size >= 20:
            processed_ids += 1
            selected_centroids.append(centroids[label][0].data.cpu().numpy())

    print(np.array(selected_centroids).shape)
    dists = pdist(np.array(selected_centroids), metric='cosine')
    print(dists)
    print(dists.shape)
    print(np.mean(dists))
    print(np.median(dists))


if __name__ == '__main__':
    with torch.no_grad():
        main()
