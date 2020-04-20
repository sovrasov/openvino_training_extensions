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

from .globalme import GlobalMe


class InternalWildtrack(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'wildtrack'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalWildtrack, self).__init__(root, market1501_500k, **kwargs)


class InternalAirport(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'airport'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalAirport, self).__init__(root, market1501_500k, **kwargs)


class InternalCameraTampering(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'camera_tampering'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalCameraTampering, self).__init__(root, market1501_500k, **kwargs)


class InternalGlobalMe(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'globalme'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalGlobalMe, self).__init__(root, market1501_500k, **kwargs)


class InternalMall(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'mall'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalMall, self).__init__(root, market1501_500k, **kwargs)


class InternalPSVIndoor(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'psv_indoor'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalPSVIndoor, self).__init__(root, market1501_500k, **kwargs)


class InternalPSVOutdoor(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'psv_outdoor'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalPSVOutdoor, self).__init__(root, market1501_500k, **kwargs)


class InternalSSPlatform(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'ss_platform'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalSSPlatform, self).__init__(root, market1501_500k, **kwargs)


class InternalSSStreet(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'ss_street'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalSSStreet, self).__init__(root, market1501_500k, **kwargs)


class InternalSSTicket(GlobalMe):
    dataset_dir = 'internal'
    dataset_subdir = 'ss_ticket'

    def __init__(self, root='', market1501_500k=False, **kwargs):
        super(InternalSSTicket, self).__init__(root, market1501_500k, **kwargs)
