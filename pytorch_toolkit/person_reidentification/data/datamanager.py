"""
 Copyright (c) 2018 Kaiyang Zhou

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

import torch

from torchreid.data.datamanager import DataManager
from torchreid.data.datasets import __image_datasets

from .datasets.chandler import Chandler
from .datasets.globalme import *
from .datasets.market1501 import Market1501
from .datasets.dukemtmcreid import DukeMTMCreID
from .datasets.msmt17 import MSMT17
from .datasets.internal import *
#from .datasets.wildtrack import Wildtrack
from .transforms import build_transforms
from .sampler import build_train_sampler

__image_datasets['msmt17'] = MSMT17
__image_datasets['market1501'] = Market1501
__image_datasets['dukemtmcreid'] = DukeMTMCreID
__image_datasets['globalme'] = GlobalMe
__image_datasets['globalmev2'] = GlobalMeV2
__image_datasets['globalmev3'] = GlobalMeV3
__image_datasets['globalmev4'] = GlobalMeV4
__image_datasets['chandler'] = Chandler
__image_datasets['wildtrack'] = Wildtrack
__image_datasets['amg'] = AMG
__image_datasets['shoppertrack'] = Shoppertrack
__image_datasets['shoppertrack-test-large'] = ShoppertrackTestLarge
__image_datasets['shoppertrack-test-small'] = ShoppertrackTestSmall
__image_datasets['amg-front'] = AMGFront
__image_datasets['amg-back'] = AMGBack
__image_datasets['amg-side'] = AMGSide
__image_datasets['psv-outdoor'] = PSVOutdoor
__image_datasets['psv-indoor'] = PSVIndoor
__image_datasets['market-train'] = MarketTrainOnly

__image_datasets['int-airport'] = InternalAirport
__image_datasets['int-camera-tampering'] = InternalCameraTampering
__image_datasets['int-globalme'] = InternalGlobalMe
__image_datasets['int-mall'] = InternalMall
__image_datasets['int-psv-indoor'] = InternalPSVIndoor
__image_datasets['int-psv-outdoor'] = InternalPSVOutdoor
__image_datasets['int-ss-platform'] = InternalSSPlatform
__image_datasets['int-ss-street'] = InternalSSStreet
__image_datasets['int-ss-ticket'] = InternalSSTicket
__image_datasets['int-wildtrack'] = InternalWildtrack


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError('Invalid dataset name. Received "{}", '
                         'but expected to be one of {}'.format(name, avai_datasets))
    return __image_datasets[name](**kwargs)


class ImageDataManagerWithTransforms(DataManager):
    data_type = 'image'

    def __init__(self, root='', sources=None, targets=None, height=256, width=128, transforms='random_flip',
                 norm_mean=None, norm_std=None, use_gpu=True, split_id=0, combineall=False,
                 batch_size_train=32, batch_size_test=32, workers=4, num_instances=4, train_sampler='',
                 cuhk03_labeled=False, cuhk03_classic_split=False, market1501_500k=False, apply_masks_to_test=False,
                 min_id_samples=0, ignore_list_path=''):
        super(ImageDataManagerWithTransforms, self).__init__(
            sources=sources, targets=targets, height=height, width=width,
            transforms=None, norm_mean=norm_mean, norm_std=norm_std, use_gpu=use_gpu
        )

        self.transform_tr, self.transform_te = build_transforms(
            self.height, self.width, transforms=transforms,
            norm_mean=norm_mean, norm_std=norm_std,
            apply_masks_to_test=apply_masks_to_test
        )

        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                min_id_samples=min_id_samples,
                ignore_list_path=ignore_list_path
            )
            trainset.append(trainset_)
        trainset = sum(trainset)
        self.trainset = trainset
        self.num_sources = len(self.sources)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train, train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False
        )

        print('=> Loading test (target) dataset')
        self.testloader = {name: {'query': None, 'gallery': None} for name in self.targets}
        self.testdataset = {name: {'query': None, 'gallery': None} for name in self.targets}

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.testloader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.testloader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset[name]['query'] = queryset.query
            self.testdataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train            : {}'.format(self.sources))
        print('  # train datasets : {}'.format(len(self.sources)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(trainset)))
        print('  # train cameras  : {}'.format(self.num_train_cams))
        print('  test             : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')
