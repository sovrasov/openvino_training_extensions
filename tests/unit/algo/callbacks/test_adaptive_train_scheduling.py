# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from unittest.mock import MagicMock

from lightning import LightningModule, Trainer
from otx.algo.callbacks.adaptive_train_scheduling import AdaptiveTrainScheduling
from torch.utils.data import DataLoader


class TestIterationTimer:
    def test_callback(self, caplog) -> None:
        callback = AdaptiveTrainScheduling(max_interval=5, decay=-0.025)

        mock_trainer = MagicMock(spec=Trainer)
        mock_pl_module = MagicMock(spec=LightningModule)

        mock_dataloader = MagicMock(spec=DataLoader)
        mock_dataloader.__len__.return_value = 10
        mock_trainer.train_dataloader = mock_dataloader

        mock_trainer.max_epochs = 10
        mock_trainer.check_val_every_n_epoch = 1
        mock_trainer.log_every_n_steps = 50

        with caplog.at_level(log.WARNING):
            callback.on_train_start(trainer=mock_trainer, pl_module=mock_pl_module)
            assert mock_trainer.check_val_every_n_epoch != 1  # Adaptively updated
            assert mock_trainer.log_every_n_steps == 10  # Equal to len(train_dataloader)
            assert len(caplog.records) == 2  # Warning two times

        callback.on_train_end(trainer=mock_trainer, pl_module=mock_pl_module)
        # Restore temporarily updated values
        assert mock_trainer.check_val_every_n_epoch == 1
        assert mock_trainer.log_every_n_steps == 50