"""
Author: TMJ
Date: 2025-01-10 16:00:59
LastEditors: TMJ
LastEditTime: 2025-01-10 16:01:37
Description: 请填写简介
"""

import warnings
from typing import Iterable, Sequence

from chemprop.data import BatchMolGraph
from chemprop.nn.hparams import HasHParams
from chemprop.nn.message_passing.proto import MessagePassing
from torch import Tensor, nn


class HybridMessagePassing(nn.Module, HasHParams):
    def __init__(
        self,
        cgr_message_passing: MessagePassing,
        additive_message_passings: Sequence[MessagePassing],
        n_components: int,
        shared: bool = False,
    ):
        super().__init__()
        self.hparams = {
            "cls": self.__class__,
            "cgr_message_passing": cgr_message_passing.hparams,
            "additive_message_passings": [
                mp.hparams for mp in additive_message_passings
            ],
            "n_components": n_components,
            "shared": shared,
        }

        if len(additive_message_passings) == 0:
            raise ValueError("At least one additive message passing must be provided")
        if shared and len(additive_message_passings) > 1:
            warnings.warn(
                "More than 1 additive message passing was supplied but"
                " 'shared' was True! Using only the 0th block..."
            )
        elif not shared and len(additive_message_passings) != n_components:
            raise ValueError(
                f"Number of additive message passings ({len(additive_message_passings)})"
                f" must match the number of components ({n_components})"
            )

        self.cgr_message_passing = cgr_message_passing
        self.n_components = n_components
        self.shared = shared
        self.additive_message_passings = nn.ModuleList(
            [additive_message_passings[0]] * self.n_components
            if shared
            else additive_message_passings
        )
    
    # TODO