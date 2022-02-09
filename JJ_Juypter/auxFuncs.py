from IPython.display import display, Math, Latex, HTML
import PIL.Image as Image
import ipywidgets as widgets
from ipywidgets import FloatSlider, ColorPicker, VBox, jslink
import ipyvolume as ipv
import os, sys, subprocess, pickle
from time import sleep
from tqdm import tqdm
from tqdm.notebook import trange
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy import stats
from scipy.linalg import hadamard
from datetime import datetime
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_precision_recall_curve, plot_roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_boston, make_classification
from sklearn.utils import resample
from sympy import solve_linear_system, Integral, sqrt 
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt, DotProduct
from collections import Counter
import imblearn
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline
import time
import random
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import gym 
from gym import Env, Wrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
from gym.envs.classic_control import rendering
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# def pdf(url):
#     return HTML('<embed src="%s" type="application/pdf" width="100%%" height="600px" />' % url)
# pdf('/Users/jordan/OneDrive/Academics/Fall2020/AI07550/MiniProject/Beamer.miniProj.pdf')
