import os
import sys
if os.path.abspath('../src') not in sys.path:
    sys.path.append(os.path.abspath('../src'))
import pandas as pd

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import GNN
from easydict import EasyDict
import time
from tqdm.auto import tqdm
from collections import defaultdict
import pysmiles
import pandas as pd
import dgl
from dgl.dataloading import GraphDataLoader
from property_pred.pp_data_processing import PropertyPredDataset
from data_processing import SmilesDataset, preprocess, get_feature_encoder, networkx_to_dgl

args = EasyDict({'pretrained_model': 'tag_1024', 'batch_size': 1024, 'gpu': '0', 'dataset': 'USPTO-479k'})

feature_encoder, train_graphs, valid_graphs, test_graphs = preprocess(args.dataset)
# train_dataset = SmilesDataset(args, 'train', feature_encoder, train_graphs)