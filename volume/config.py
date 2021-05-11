"""
Configuration file for Blender environment

Contains constants that will be used for Blender master project

Written by Naphat Amundsen
"""

from numpy import pi

# There are names of Blender objects in this file, the Blender must manually be set to corresponding
# names

"""generate.py"""
SRC_CLTN = "Originals"  # Collection of original objects
TRGT_CLTN = "Copies"  # Collection of copies (to be rendered)
REF_CLTN = "Reference"  # Collection of reference item, used to sanity check renders
CAM_CLTN = "Cameras"  # Collection of camera objects
SPAWNBOX_OBJ = "spawnbox"  # Spawnbox object, representing the spawn area
CAMERA_OBJ_CENTER = "camera_C"  # Name of center camera object, should be same name in Blender file
CAMERA_OBJ_CENTER_TOP = "camera_C_TOP"  # Name of center camera object, should be same name in Blender file
CAMERA_OBJ_LEFT = "camera_L"  # Name of left camera object, should be same name in Blender file
CAMERA_OBJ_RIGHT = "camera_R"  # Name of right camera object, should be same name in Blender file
ROT_MUS = [0, 0, -0.5*pi]  # Mean rotation for fishes when generating
ROT_STDS = [2*pi, 2*pi, 2*pi]  # Std rotation for fishen when generating
DEFAULT_BBOX_MODE = "full"  # cps xyz full std
COMMIT_INTERVAL = 32  # How often to commit to database (16 means commit at every 16th sample)
RAND_SCALE_MU: float = 1
RAND_SCALE_STD: float = 0.2
CLASS_DICT = {  # Enforce class dictionary, inverse map: {v: k for k, v in CLASS_DICT.items()}
    "haddock": 0,
    "hake": 1,
    "herring": 2,
    "mackerel": 3,
    "redgurnard": 4,
    "whiting": 5,
}
DEFAULT_SPAWNRANGE = (1, 6) # Draw from uniform dist (1, 6) to determine how many fish to spawn

"""reconstruct.py"""
DEFAULT_ALTER_COLOR = (0.2, 1, 0.2, 1) # R G B A
DEFAULT_MIXSHADER_FAC = 0.3 # Mixing between transparent and texture

"""Filesystem, data and database"""
GENERATED_DATA_DIR = "generated_data"
# BBOX_FILE = 'bboxes.csv'
IMAGE_DIR = "images"  # Will be placed in GENERATED_DATA_DIR
IMAGE_NAME = "img"
DEFAULT_FILEFORMAT = "PNG"  # This is what you give to Blender, the actual file extension is:
DEFAULT_FILEFORMAT_EXTENSION = ".png"  # The actual file extension in file system
BBOX_DB_FILE = "bboxes.db"
METADATA_FILE = "metadata.txt"
BBOX_DB_IMGRNR = "imgnr"  # Column name for image id
BBOX_DB_CLASS = "class_"  # Column name for classes
BBOX_MODE_CPS = "cps"  # Cornerponts
BBOX_MODE_XYZ = "xyz"  # Lengths in x, y, z dimension
BBOX_MODE_FULL = "full"  # Lengths in x, y, z dimension
BBOX_MODE_STD = "std"  # Standard 2D bbox for object detection
BBOX_DB_TABLE_XYZ = "bboxes_xyz"  # Length width height
BBOX_DB_TABLE_CPS = "bboxes_cps"  # Corner points
BBOX_DB_TABLE_STD = "bboxes_std"  # Standard bounding boxes
BBOX_DB_TABLE_FULL = "bboxes_full"  # Standard bounding boxes
FILE_SUFFIX_CENTER = "_C"
FILE_SUFFIX_LEFT = "_L"  # Rendering only from one direction will not generate file suffixes
FILE_SUFFIX_RIGHT = "_R"
FILE_SUFFIX_CENTER_TOP = "_C_TOP"
RENDER_RES_X = 416 # Render res is atm or documentation only, the code wont use it atm
RENDER_RES_Y = 416 # Render res is atm or documentation only, the code wont use it atm

LABELCHECK_DATA_DIR = "label_renders" # "root" directory for labelchekc output
LABELCHECK_IMAGE_DIR = "reconstructed_labels"  # Will be placed in LABELCHECK_DATA_DIR
LABELCHECK_IMAGE_NAME = "reimg"
LABELCHECK_DB_TABLE = "rendered" 
LABELCHECK_DB_IMGNR = "imgnr" 
LABELCHECK_DB_FILE = "recon.db"

"""CLI"""
# options suffixed with _SHORT are the shortened version of the big one
# none of the OPT_* stuff is used in code as of 09/01/2021
OPT_ENGINE = "--engine"  # Choose engine, (BLENDER_EEVEE, CYCLES)
OPT_ENGINE_SHORT = "-e"
OPT_CLEAR = "--clear"
OPT_CLEAR_EXIT = "--clear-exit"
OPT_DEVICE = "--device"
OPT_DEVICE_SHORT = "--d"
OPT_SAMPLES = "--samples"
OPT_SAMPLES_SHORT = "-s"
OPT_BBOX = "--bbox"
OPT_BBOX_SHORT = "-b"
OPT_REFERENCE = "--reference"
OPT_REFERENCE_SHORT = "-r"
OPT_VIEW_MODE = "--view-mode"
OPT_NO_WAIT = "--no-wait"
OPT_DIR = "--dir"
OPT_STDBBOXCAM = "--stdbboxcam"

ARGS_DEFAULT_ENGINE = "CYCLES"  # [BLENDER_EEVEE, CYCLES]
ARGS_DEFAULT_DEVICE = "CUDA"
ARGS_DEFAULT_RENDER_SAMPLES = 96
ARGS_DEFAULT_BBOX_MODE = "all"  # [BBOX_MODE_CPS, BBOX_MODE_XYZ, 'all']
ARGS_DEFAULT_VIEW_MODE = "leftright"  # [leftright, center, topside]
ARGS_DEFAULT_STDBBOX_CAM = "left"


"""INFO"""
HIGHLIGHT_MIN_WIDTH = 72
BOXED_SYMBOL_TOP = "="  # Only single char
BOXED_SYMBOL_BOTTOM = "="  # Only single char
BOXED_STR_SIDE = "||"  # Can be string
SECTION_SYMBOL = "-"  # Only single char
SECTION_START_STR = ""
SECTION_END_STR = "\n"
