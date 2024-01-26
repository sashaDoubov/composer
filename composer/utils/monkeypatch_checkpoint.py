import collections
import dataclasses
import io
import os
import pickle
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch._utils import _get_device_module
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._checkpointer import _Checkpointer
from torch.futures import Future
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter, WriteResult
from torch.distributed.checkpoint.filesystem import FileSystemWriter, _split_by_size_and_type, DEFAULT_SUFFIX, _write_files_from_queue

class MonkeyPathFileSystemWriter(FileSystemWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        if not isinstance(path, Path):
            path = Path(path)

        
        self.per_rank_path = path

        self.rank_monkeypatch = False
        if 'RANK' in str(path):
            self.rank_monkeypatch = True
            path = path.parent
        
        self.path = path
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.thread_count = thread_count
        self.per_thread_copy_ahead = per_thread_copy_ahead


    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.per_rank_path.mkdir(parents=True, exist_ok=True)
        return plan

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[List[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        file_queue: queue.Queue = queue.Queue()
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(self.thread_count, plan.items):
                file_name = gen_file()
                if self.rank_monkeypatch:
                    file_queue.put((self.per_rank_path / file_name, os.path.join(self.per_rank_path.name, file_name), bucket))
                else:
                    file_queue.put((self.path / file_name, file_name, bucket))
        else:
            for item in plan.items:
                file_name = gen_file()
                if self.rank_monkeypatch:
                    file_queue.put((self.per_rank_path / file_name, os.path.join(self.per_rank_path.name, file_name), [item]))
                else:
                    file_queue.put((self.path / file_name, file_name, [item]))

        result_queue: queue.Queue = queue.Queue()

        threads = []
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=_write_files_from_queue,
                args=(
                    file_queue,
                    result_queue,
                    planner,
                    self.per_thread_copy_ahead,
                    self.sync_files,
                ),
            )
            t.start()
            threads.append(t)

        _write_files_from_queue(
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            inflight_threshhold=self.per_thread_copy_ahead,
            use_fsync=self.sync_files,
        )

        for t in threads:
            t.join()

        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            pass

            fut: Future[List[WriteResult]] = Future()
            fut.set_result(res)
            return fut