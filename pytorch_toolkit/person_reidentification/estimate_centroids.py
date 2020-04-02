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
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, cdist

import torch.nn as nn
import torch.nn.functional as F

from torchreid.utils import load_pretrained_weights
from config.default_config import get_default_config, imagedata_kwargs
from models.builder import build_model
from data.datamanager import ImageDataManagerWithTransforms


def sort_dict(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: -item[1])}


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--ids-file', type=str, default='ids_to_remove.json')
    parser.add_argument('--cut-ids-percent', type=int, default=3)
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
            pooling_type=cfg.model.pooling_type,
            input_size=(cfg.data.height, cfg.data.width),
            dropout_cfg=cfg.model.dropout,
            IN_first=cfg.model.IN_first,
            extra_blocks=cfg.model.extra_blocks,
            lct_gate=cfg.model.lct_gate
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
    nrof_images = len(datamanager.trainset)
    emb_array = np.zeros((nrof_images, cfg.model.feature_dim), dtype=np.float32)
    pids_array = np.zeros(nrof_images, dtype=np.int32)
    for j, item in enumerate(tqdm(loader)):
        embeddings = F.normalize(model(item[0]), dim=1)
        # 0 - images 1 - labels 2 - cams 3 - paths
        start_index = j * cfg.train.batch_size
        end_index = min((j + 1) * cfg.train.batch_size, nrof_images)
        batch_embeddings = embeddings.data.cpu().numpy()
        assert emb_array[start_index:end_index, :].shape == batch_embeddings.shape
        emb_array[start_index:end_index, :] = batch_embeddings
        pids_array[start_index:end_index] = item[1].data.cpu().numpy()

    weak_criterion = {}
    strong_criterion = {}
    for pid in set(pids_array):
        pid_embs = emb_array[np.where(pids_array == pid)]
        center = np.mean(pid_embs, axis=0)
        centroids[pid] = [center, pid_embs.shape[0]]
        pairwise_dists = cdist(pid_embs, pid_embs, metric='cosine')
        pairwise_dists_center = cdist([center], pid_embs, metric='cosine')
        weak_criterion[pid] = np.max(pairwise_dists_center)
        strong_criterion[pid] = np.max(pairwise_dists)

    weak_criterion = sort_dict(weak_criterion)
    strong_criterion = sort_dict(strong_criterion)
    cut_len = int(args.cut_ids_percent / 100 * len(weak_criterion)) + 1
    weak_ids_cut = list(weak_criterion.keys())[0:cut_len]
    strong_ids_cut = list(strong_criterion.keys())[0:cut_len]
    print(strong_ids_cut)
    print(weak_ids_cut)
    verified_set = set(weak_ids_cut).intersection(set(strong_ids_cut))
    print(sorted(verified_set))

    removed_count = 0
    removed_names = []
    for item in datamanager.trainset.data:
        if item[1] in verified_set:
            removed_count += 1
            removed_names.append(os.path.split(item[0])[1])

    print('Images removed ' + str(removed_count))

    processed_ids = 0
    max_processed = 200
    selected_centroids = []
    for label in centroids:
        if processed_ids >= max_processed:
            break
        size = centroids[label][1]
        if size >= 20:
            processed_ids += 1
            selected_centroids.append(centroids[label][0])

    with open(args.ids_file, 'w') as outfile:
        json.dump(removed_names, outfile)

    dists = pdist(np.array(selected_centroids), metric='cosine')
    print(dists)
    print(dists.shape)
    print(np.mean(dists))
    print(np.median(dists))


if __name__ == '__main__':
    with torch.no_grad():
        main()
