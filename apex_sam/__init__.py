from apex_sam.config import ApexConfig

__all__ = ["ApexConfig", "ApexSegmenter", "FileSystemSupportPool", "LocalSupportDB", "run_evaluation"]


def __getattr__(name: str):
    if name == "ApexSegmenter":
        from apex_sam.pipeline.segmenter import ApexSegmenter

        return ApexSegmenter
    if name == "FileSystemSupportPool":
        from apex_sam.support.filesystem_pool import FileSystemSupportPool

        return FileSystemSupportPool
    if name == "LocalSupportDB":
        from apex_sam.retrieval.local_db import LocalSupportDB

        return LocalSupportDB
    if name == "run_evaluation":
        from apex_sam.evaluation.runner import run_evaluation

        return run_evaluation
    raise AttributeError(name)
