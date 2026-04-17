from __future__ import annotations

from typing import Protocol

import numpy as np

from apex_sam.types import SupportMatch


class SupportProvider(Protocol):
    """Public retrieval interface exposed by the open-source release."""

    def search(
        self,
        query_image: np.ndarray,
        *,
        label_value: int,
        topk: int,
        exclude_case_id: str | None = None,
        exclude_slice_index: int | None = None,
    ) -> list[SupportMatch]:
        ...
