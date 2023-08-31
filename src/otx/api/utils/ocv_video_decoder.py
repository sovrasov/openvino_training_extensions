import atexit
import logging
import os
import abc
from multiprocessing import Pool, current_process
from typing import TYPE_CHECKING, NamedTuple, Union
import threading
from typing import Any

import cv2
import numpy as np

#if TYPE_CHECKING:
import multiprocessing

logger = logging.getLogger(__name__)


DISABLE_VIDEO_DECODING_MULTIPROC = (
    os.getenv("DISABLE_VIDEO_DECODING_MULTIPROC", "0") == "1"
)
DISABLE_VIDEO_DECODING_MULTIPROC = False



class _FrameCacheOpenCV(NamedTuple):
    file_location: str
    frame_index: int
    video_capture: cv2.VideoCapture
    frame: np.ndarray

class _VideoInformation(NamedTuple):
    fps: float
    width: int
    height: int
    total_frames: int


class TimeoutLock:
    """
    Wrapper around threading.Lock or threading.RLock to use the timeout functionality in
    a context manager.

    :param timeout: Timeout in seconds
    :param use_rlock: boolean if RLock should be used instead of a Lock

    Usage:
        >>> lock = TimeoutLock(timeout=3)
        >>> try:
        >>>     with lock:
        >>>         ...  # success
        >>> except TimeoutError:
        >>>     ...      # failure
    """

    def __init__(self, timeout: float, use_rlock: bool = False) -> None:
        self._lock: Union[threading.Lock, threading.RLock]
        if use_rlock:
            self._lock = threading.RLock()
        else:
            self._lock = threading.Lock()
        self._timeout = timeout

    def acquire(self):
        """Acquire the lock explicitly"""
        return self._lock.acquire(timeout=self._timeout)

    def release(self):
        """Release the lock explicitly"""
        self._lock.release()

    def __enter__(self):
        """
        Called when entering the context.

        :return: the acquired threading.Lock object
        :raises TimeoutError: if the lock cannot be acquired within timeout
        """
        lock_acquired = self._lock.acquire(timeout=self._timeout)
        if lock_acquired:
            return self._lock
        raise TimeoutError(f"Failed to acquire lock within timeout ({self._timeout})")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when leaving the context"""
        self._lock.release()


class Singleton(abc.ABCMeta):
    # We subclass GenericMeta instead of ABCMeta to allow Singleton to be used as metaclass for a generic class.
    """
    Metaclass to create singleton classes.

    :example: The class can be used as follows:

        .. code-block:: python
            class Foo(metaclass=Singleton):
                def __init__(self):
                    ...

        The class `Foo` will be instantiated only once and subsequent calls to `Foo()`
        will return the existing instance of `Foo`.
    """

    def __new__(cls, clsname, bases, dct):
        def __deepcopy__(self, memo_dict=None):
            # Since the classes are singleton per-project, we can return the same object.
            if memo_dict is None:
                memo_dict = {}
            memo_dict[id(self)] = self
            return self

        def __copy__(self):
            # Since the classes are singleton per-project, we can return the same object.
            return self

        newclass = super().__new__(cls, clsname, bases, dct)
        setattr(newclass, __deepcopy__.__name__, __deepcopy__)
        setattr(newclass, __copy__.__name__, __copy__)
        return newclass

    def __init__(cls, clsname, bases, dct):
        super().__init__(clsname, bases, dct)
        cls._lock = threading.Lock()
        cls._instance: Union[Singleton, None] = None

    def __call__(cls, *args, **kwargs) -> Any:
        if cls._instance is None:  # double-checked locking
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__call__(*args, **kwargs)
        return cls._instance



class _VideoDecoderOpenCV(metaclass=Singleton):
    """
    Video decoder.

    This class wraps the opencv video capture. However, video decoding is typically multithreaded which can cause
    segmentation faults when used in conjunction with the forking multiprocessing commonly used in data loaders.
    This is because C/C++ multithreading data structures are not properly cleaned up in the forked process.
    To avoid this problem class does the decoding in a separate worker process if called from a non-daemonic
    process (i.e. the main process). Daemonic processes are safe because they are not allowed to spawn processes.

    This class also caches the last video and frame index used to improve the performance of sequential reads.
    """

    __frame_cache_numpy: Union[_FrameCacheOpenCV, None] = None
    __frame_cache_numpy_lock = TimeoutLock(10)

    @staticmethod
    def _get_video_information(file_location: str) -> _VideoInformation:
        """
        Create a _VideoInformation object with descriptive information about the video

        :param file_location: Local path or presigned S3 URL pointing to the video
        :return: _VideoInformation object containing information about the video
        """
        with _VideoDecoderOpenCV.__frame_cache_numpy_lock:
            video_capture = cv2.VideoCapture(file_location)
            info = _VideoInformation(
                float(video_capture.get(cv2.CAP_PROP_FPS)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            )

            _VideoDecoderOpenCV.__frame_cache_numpy = _FrameCacheOpenCV(
                file_location=file_location,
                frame_index=-1,
                video_capture=video_capture,
                frame=np.array(()),
            )

            return info

    @staticmethod
    def _decode(file_location: str, frame_index: int) -> np.ndarray:
        """
        Decode the video and return the requested frame in array format

        :param file_location: Local storage path or presigned S3 URL pointing to the video
        :param frame_index: Frame index for the requested frame
        :return: Numpy array for the requested frame
        """
        with _VideoDecoderOpenCV.__frame_cache_numpy_lock:
            cached_value = _VideoDecoderOpenCV.__frame_cache_numpy

            if cached_value is not None and cached_value.file_location == file_location:
                last_frame_index = cached_value.frame_index
                video_capture = cached_value.video_capture
                last_frame = cached_value.frame

                if frame_index == last_frame_index:
                    return last_frame
                if frame_index != last_frame_index + 1:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            else:
                # Release cached video_capture first
                if cached_value is not None:
                    cached_value.video_capture.release()
                # set frame position
                video_capture = cv2.VideoCapture(file_location)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            actual_frame_index = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))

            if actual_frame_index != frame_index:
                print(
                    f"Warning: Got frame {actual_frame_index}, while asking for {frame_index}"
                )

            success, frame = video_capture.read()

            if not success:
                # Issue in opencv video capture can cause the index to be off. Try resetting the capture and read again.
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = video_capture.read()
                if not success:
                    raise ValueError(
                        f"Unable to read frame `{frame_index}` of video file located at {file_location}"
                    )

            result_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            _VideoDecoderOpenCV.__frame_cache_numpy = _FrameCacheOpenCV(
                file_location, frame_index, video_capture, result_frame
            )

            return result_frame

    def __init__(self) -> None:
        # The multiprocessing module has a pool.Pool class which is created by a Pool() function.
        # Despite having the same name, they are two different things.
        self._process: Union[multiprocessing.pool.Pool, None] = None
        self._pid = -1

        atexit.register(self.close)

    def close(self) -> None:
        """
        Close self._process if possible.
        """
        if self._process is not None:
            self._process.close()

    def _get_process(self) -> Union[multiprocessing.pool.Pool, None]:
        """
        Return the worker process if any.

        If the current process is a daemon, we return ``None`` and work directly in the current process since pools and
        processes cannot be spawned from a daemon process. However if the process is not a daemon, we return
        a worker process.
        """
        pid = os.getpid()

        # Check if the process was forked (pid change)
        if self._pid != pid:
            # New process. The old self._process will not be valid anymore and need to be recreated.
            # Check whether the new process is a daemon or not to determine whether to use a worker process
            # or instead do the processing in the current process.
            if current_process().daemon:
                self._process = None
            else:
                self._process = Pool(1)
            self._pid = pid
        return self._process

    def _execute(self, func, **kwargs):
        """
        Execute a function with multiprocessing or not, this depends on the OS
        environment variable DISABLE_VIDEO_DECODING_MULTIPROC (0|1).

        If DISABLE_VIDEO_DECODING_MULTIPROC is 1, run the function with a single thread,
        otherwise run with multiprocessing

        :param func: The function to be executed
        :param kwargs: The kwargs to pass onto the function f
        :return: The return value of the function f
        """
        if os.getenv("DISABLE_VIDEO_DECODING_MULTIPROC", "0") == "1":
            process = None
            # Multiprocessing is disabled for the resource and director microservices,
            # because disabling it is faster. For other tasks it is enabled, to avoid
            # conflicts between FFmpeg and PyTorch.
        else:
            process = self._get_process()
        if process is None:
            return func(**kwargs)
        return process.apply(func, kwds=kwargs)

    def get_video_information(self, file_location: str) -> _VideoInformation:
        """
        Retrieve basic information about a given video.

        :param file_location: Local storage path or presigned URL pointing to the video
        :return: VideoInformation object describing the video information
        """
        kwargs = {"file_location": file_location}
        return self._execute(func=_VideoDecoderOpenCV._get_video_information, **kwargs)

    def decode(self, file_location: str, frame_index: int) -> np.ndarray:
        """
        Decode and return a frame from the video.

        :param file_location: Local storage path or presigned S3 url pointing to the video
        :param frame_index: Index of the frame in the video
        :return: The frame in RGB format
        """
        kwargs = {"file_location": file_location, "frame_index": frame_index}
        return self._execute(func=_VideoDecoderOpenCV._decode, **kwargs)