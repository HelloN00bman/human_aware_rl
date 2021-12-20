import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.datasets import MNIST
from torchvision.utils import save_image

import numpy as np
# %matplotlib inline
# %matplotlib notebook
import tqdm, copy
import random, os
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
from collections import defaultdict
from overcooked_ai_py.utils import save_pickle
from human_aware_rl.utils import set_global_seed
# from human_aware_rl.human.process_dataframes import *
# from human_aware_rl.static import *
import pickle as pkl

set_global_seed(1884)
import ast
import json

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
from ast import literal_eval
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

NORTH = (0, -1)
SOUTH = (0, 1)
EAST = (1, 0)
WEST = (-1, 0)
STAY = (0, 0)
INTERACT = 'INTERACT'



