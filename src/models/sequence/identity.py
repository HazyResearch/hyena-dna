from .base import SequenceModule


class Identity(SequenceModule):
    _name_ = 'identity'
    _d_model = 1
    _d_output = 1
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()