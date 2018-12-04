import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem

from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
#parser.add_option("-t", "--test", dest="test_path")
#parser.add_option("-v", "--vocab", dest="vocab_path")
#parser.add_option("-m", "--model", dest="model_path")
#parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
#parser.add_option("-l", "--latent", dest="latent_size", default=56)
#parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open('../data/vocab.txt')] 
vocab = Vocab(vocab)
print(type(vocab))
hidden_size = 450
latent_size = 56
depth = 3

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load('MPNVAE-h450-L56-d3-beta0.005/model.iter-4',map_location=lambda storage, loc: storage))
#model = model.cuda()

data = []
with open('../data/test.txt') as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

acc = 0.0
tot = 0
for smiles in data:
    mol = Chem.MolFromSmiles(smiles)
    smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
    for i in xrange(10):

	    dec_smiles = model.reconstruct(smiles3D)
	    print(dec_smiles)
   # print(dec_smiles)
#    print(smiles3D)
#    if dec_smiles == smiles3D:
 #       acc += 1
  #  tot += 1
   # print acc / tot

#   " acc=0.0
#    tot=0
#    dec_smiles = model.recon_eval(smiles3D)
#    tot += len(dec_smiles)
#    for s in dec_smiles:
#        if s == smiles3D:
#            acc += 1
#    print acc / tot"
    

