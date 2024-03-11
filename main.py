from train import train_model
from ssa import SSA
import numpy as np

lb = [0, 0, 0, 0, 0, 0, 0, 0, 0]
ub = [0, 0, 0, 0, 0, 0, 0, 0, 0]
lb = np.array(lb)[None, :]
ub = np.array(ub)[None, :]
fMin, bestX = SSA(lb, ub, lb.shape[-1], train_model)




