from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()

_C.MODEL.WEIGHT = ""
_C.MODEL.WEIGHT_50 = "1fZYAesxzlXrHIz_3uFRcVmB0c_jPf-GC"
_C.MODEL.WEIGHT_101 = "1AtK3losIFJ_vuRwRry4K2eoXnhBEkVSZ"

# -----------------------------------------------------------------------------
# ROI action head config.
# -----------------------------------------------------------------------------
_C.MODEL.ROI_ACTION_HEAD = CN()


_C.MODEL.ROI_ACTION_HEAD.POOLER_RESOLUTION = 7
_C.MODEL.ROI_ACTION_HEAD.POOLER_SCALE = 1./16
# Only used for align3d
_C.MODEL.ROI_ACTION_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER = True

_C.MODEL.ROI_ACTION_HEAD.MLP_HEAD_DIM = 1024

_C.MODEL.ROI_ACTION_HEAD.DROPOUT_RATE = 0.0
_C.MODEL.ROI_ACTION_HEAD.NUM_CLASSES = 80

# Action loss evaluator config.
_C.MODEL.ROI_ACTION_HEAD.PROPOSAL_PER_CLIP = 10
_C.MODEL.ROI_ACTION_HEAD.NUM_PERSON_MOVEMENT_CLASSES = 14
_C.MODEL.ROI_ACTION_HEAD.NUM_OBJECT_MANIPULATION_CLASSES = 49
_C.MODEL.ROI_ACTION_HEAD.NUM_PERSON_INTERACTION_CLASSES = 17


_C.IA_STRUCTURE = CN()
_C.IA_STRUCTURE.ACTIVE = True
_C.IA_STRUCTURE.STRUCTURE = "dense"
_C.IA_STRUCTURE.MAX_PER_SEC = 5
_C.IA_STRUCTURE.MAX_PERSON = 25
_C.IA_STRUCTURE.DIM_IN = 2304
_C.IA_STRUCTURE.DIM_INNER = 1024
_C.IA_STRUCTURE.DIM_OUT = 2304
_C.IA_STRUCTURE.LENGTH = (30, 30)
_C.IA_STRUCTURE.MEMORY_RATE = 1
_C.IA_STRUCTURE.FUSION = "add"
_C.IA_STRUCTURE.CONV_INIT_STD = 0.01
_C.IA_STRUCTURE.DROPOUT = 0.
_C.IA_STRUCTURE.NO_BIAS = False
_C.IA_STRUCTURE.I_BLOCK_LIST = ['P', 'O', 'M', 'P', 'O', 'M']
_C.IA_STRUCTURE.LAYER_NORM = True
_C.IA_STRUCTURE.TEMPORAL_POSITION = True
_C.IA_STRUCTURE.ROI_DIM_REDUCE = True
_C.IA_STRUCTURE.USE_ZERO_INIT_CONV = True
_C.IA_STRUCTURE.MAX_OBJECT = 5

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Config used in inference.
_C.TEST.EXTEND_SCALE = (0.1, 0.05)