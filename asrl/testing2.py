# Import required packages
import torch
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt

from tqdm import tqdm
base = nf.distributions.base.DiagGaussian(2)

conv = nf.flows.Conv1dCouplingBlock(n_channels, 64, 2, 3, 1)