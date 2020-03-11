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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import glob
import re
import warnings

from torchreid.data.datasets import ImageDataset


class GlobalMe(ImageDataset):
    """GlobalMe.

    Dataset statistics:
        - identities: 1610.
        - images: 0 (train) + 8450 (query) + 41107 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'globalme-reid'
    dataset_subdir = 'GlobalMe-reID'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, self.dataset_subdir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"{}".'.format(self.dataset_subdir))

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(GlobalMe, self).__init__(train, query, gallery, **kwargs)

    @staticmethod
    def process_dir(dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        return data


class GlobalMeV2(GlobalMe):
    dataset_dir = 'globalmev2-reid'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(GlobalMeV2, self).__init__(root, market1501_500k, **kwargs)


class GlobalMeV3(GlobalMe):
    dataset_dir = 'globalmev3-reid'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(GlobalMeV3, self).__init__(root, market1501_500k, **kwargs)


class GlobalMeV4(GlobalMe):
    dataset_dir = 'globalmev4-reid'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(GlobalMeV4, self).__init__(root, market1501_500k, **kwargs)


class Wildtrack(GlobalMe):
    dataset_dir = 'wildtrack-reid'
    dataset_subdir = 'Wildtrack'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(Wildtrack, self).__init__(root, market1501_500k, **kwargs)


class AMG(GlobalMe):
    dataset_dir = 'amg-reid'
    dataset_subdir = 'AMG'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(AMG, self).__init__(root, market1501_500k, **kwargs)


class ShoppertrackTestLarge(GlobalMe):
    dataset_dir = 'shoppertrack-reid'
    dataset_subdir = 'Shoppertrack-large'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(ShoppertrackTestLarge, self).__init__(root, market1501_500k, **kwargs)


class ShoppertrackTestSmall(GlobalMe):
    dataset_dir = 'shoppertrack-reid'
    dataset_subdir = 'Shoppertrack-small'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(ShoppertrackTestSmall, self).__init__(root, market1501_500k, **kwargs)


class Shoppertrack(GlobalMe):
    dataset_dir = 'shoppertrack-train-reid'
    dataset_subdir = 'Shoppertrack'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(Shoppertrack, self).__init__(root, market1501_500k, **kwargs)


class AMGFront(GlobalMe):
    dataset_dir = 'amg-reid-front'
    dataset_subdir = 'AMG'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(AMGFront, self).__init__(root, market1501_500k, **kwargs)


class AMGBack(GlobalMe):
    dataset_dir = 'amg-reid-back'
    dataset_subdir = 'AMG'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(AMGBack, self).__init__(root, market1501_500k, **kwargs)


class AMGSide(GlobalMe):
    dataset_dir = 'amg-reid-side'
    dataset_subdir = 'AMG'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(AMGSide, self).__init__(root, market1501_500k, **kwargs)


class PSVOutdoor(GlobalMe):
    dataset_dir = 'psv-outdoor-reid'
    dataset_subdir = 'PSV-Outdoor-reID'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(PSVOutdoor, self).__init__(root, market1501_500k, **kwargs)


class PSVIndoor(GlobalMe):
    dataset_dir = 'psv-indoor-reid'
    dataset_subdir = 'PSV-Indoor-reID'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(PSVIndoor, self).__init__(root, market1501_500k, **kwargs)


class MarketTrainOnly(GlobalMe):
    dataset_dir = 'market1501'
    dataset_subdir = 'Train-only'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(MarketTrainOnly, self).__init__(root, market1501_500k, **kwargs)
