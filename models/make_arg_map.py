import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from scipy.io import savemat, loadmat
from utils.utils import make_arg_map_

face_vector_img_8 = loadmat(str(ROOT_DIR / "mesh/level5_vector.mat"))["vector"]
face_vector_img_tr_8 = face_vector_img_8.reshape(-1, 3)
img_level = 5
arg_map, dist_vector = make_arg_map_(face_vector_img_8,512,1024,60,120, img_level)
