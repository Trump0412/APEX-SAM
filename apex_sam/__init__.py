from apex_sam.config import ApexConfig

__all__ = ["ApexConfig", "ApexSegmenter", "run_evaluation"]


def __getattr__(name: str):
    if name == "ApexSegmenter":
        from apex_sam.pipeline.segmenter import ApexSegmenter

        return ApexSegmenter
    if name == "run_evaluation":
        from apex_sam.evaluation.runner import run_evaluation

        return run_evaluation
    raise AttributeError(name)
