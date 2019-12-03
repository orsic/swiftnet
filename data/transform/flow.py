from .flow_utils import subsample_flow

__all__ = ['SubsampleFlow']


class SubsampleFlow:
    def __init__(self, subsampling=4):
        self.subsampling = subsampling

    def __call__(self, example):
        example['flow_sub'] = subsample_flow(example['flow'], self.subsampling)
        return example
