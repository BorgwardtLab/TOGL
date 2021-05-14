import os.path
from enum import Enum, auto
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


class Tasks(Enum):
    """Valid tasks."""

    GRAPH_CLASSIFICATION = auto()
    NODE_CLASSIFICATION = auto()
    NODE_CLASSIFICATION_WEIGHTED = auto()
#from . import topo_utils
#from . import coord_transforms
#from . import data_utils
#from . import models
