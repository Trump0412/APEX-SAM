from __future__ import annotations


class PrivateQARStub:
    """
    Placeholder for the private QAR module.

    The full QAR database construction and quality-aware dual-stream retrieval
    used in the paper are intentionally not released because the original
    medical support bank contains sensitive data.
    """

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - explicit stub
        raise NotImplementedError(
            "Private QAR is not included in the open-source release. "
            "Please use the public support-pool interface instead."
        )
