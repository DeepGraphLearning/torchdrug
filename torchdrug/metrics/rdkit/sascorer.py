import os
import sys
import math
import pickle

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from torchdrug import utils

module = sys.modules[__name__]
path = os.path.dirname(__file__)

# Calculate synthetic accessibility of molecules
# Code adapted from RDKit
# https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py


def readFragmentScores():
    url = "https://github.com/rdkit/rdkit/raw/master/Contrib/SA_Score/fpscores.pkl.gz"
    md5 = "2f80a169f9075e977154f9caec9e5c26"

    zip_file = utils.download(url, path, md5=md5)
    pkl_file = utils.extract(zip_file)
    with open(pkl_file, "rb") as fin:
        data = pickle.load(fin)
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    return outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if not hasattr(module, "fscores"):
        module.fscores = readFragmentScores()
    fscores = module.fscores

    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += fscores.get(sfp, -4) * v
    score1 /= nf

    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore