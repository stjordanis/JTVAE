import sys
        sys.path.append('/home/icml18-jtnn')
        sys.path.append('gs://tox/icml18-jtnn/')
        sys.path.append('/home/naman_churiwala_quantiphi_com/icml18-jtnn')
        import torch
        import torch.nn as nn
        from torch.autograd import Variable
        from optparse import OptionParser
        import rdkit
        from rdkit.Chem import Descriptors
        from rdkit.Chem import MolFromSmiles, MolToSmiles
        from rdkit.Chem import rdmolops
        import sascorer
        import numpy as np
        from jtnn import *
        import pickle
        import gzip
        from sparse_gp import SparseGP
        import scipy.stats as sps
        import os.path
        import time
        import sys
        import rdkit.Chem as Chem
        from rdkit.Chem import Draw
        from rdkit.Chem import Descriptors
        import sascorer
        from gen_latent_demo import gen_latent_demo
        from run_bo_demo import run_bo_demo
        from print_result_smiles import print_result_smiles
    
from inference.py import inference
var=inference()
print(var)
