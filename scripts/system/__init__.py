import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules 

import psycopg
from pgvector.psycopg import register_vector

import ollama
import streamlit as st

import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
ITEMS_CHECKPOINT_PATH = '../../checkpoints/removed-items-classification-vit/best_model_2024-04-16_03-43-03_epoch_4.pt'
OUTFIT_CHECKPOINT_PATH = '../../checkpoints/outfit-classification-mlp/best_model_2024-05-12_09-07-09_epoch_4.pt'
RULES_PATH = 'rules.csv'
INPUT_SIZE = 224
ITEM_LABEL_DICT = {
    0: 'coats_jackets',
    1: 'dresses',
    2: 'handbags',
    3: 'hats',
    4: 'jewelry',
    5: 'pants',
    6: 'scarves_shawls',
    7: 'shirts_tops',
    8: 'shoes',
    9: 'shorts',
    10: 'skirts',
    11: 'sunglasses',
    12: 'watches'
}