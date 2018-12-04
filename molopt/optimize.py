import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('/home/naman_churiwala_quantiphi_com/icml18-jtnn')
sys.path.append('/home/icml18-jtnn')
import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import sascorer

from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default='../data/test2.txt')
#parser.add_option("-v", "--vocab", dest="vocab_path")
#parser.add_option("-m", "--model", dest="model_path")
parser.add_argument('--hidden_size', type=int, default=420)
parser.add_argument( "--latent_size", type=int, default=56)
parser.add_argument( "--depth", type=int, default=3)
parser.add_argument("--cutoff", type=float, default=0.2)
args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open('../data/vocab.txt')] 
vocab = Vocab(vocab)

hidden_size = int(args.hidden_size)
latent_size = int(args.latent_size)
depth = int(args.depth)
sim_cutoff = float(args.cutoff)

model = JTPropVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load('joint-h420-L56-d3-beta0.005/model.iter-4',map_location=lambda storage, loc: storage))
#model = model.cuda()

data = []
with open(args.data_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

res = []
for smiles in data:
    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol) - sascorer.calculateScore(mol)

    new_smiles,sim = model.optimize(smiles, sim_cutoff=sim_cutoff, lr=2, num_iter=80)
    new_mol = Chem.MolFromSmiles(new_smiles)
    new_score = Descriptors.MolLogP(new_mol) - sascorer.calculateScore(new_mol)

    res.append( (new_score - score, sim, score, new_score, smiles, new_smiles) )
    print new_score - score, sim, smiles, new_smiles

print sum([x[0] for x in res])
