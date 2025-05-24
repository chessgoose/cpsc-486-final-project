# http://github.com/seyonechithrananda/bert-loves-chemistry

#!/usr/bin/env python3
import re
import random
import pandas as pd
from rdkit import Chem

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
INPUT_CSV    = 'SMILES_test.csv'      # path to your input
SMILES_COL   = 'smiles'               # name of the column with SMILES
OUTPUT_CSV   = 'masked_smiles.csv'    # where to save

MASK_TOKEN   = '[MASK]'               # ChemBERTa mask token
MASK_PROB    = 0.15                   # fallback token‐masking probability

# Functional groups to consider masking (SMARTS : name)
FG_SMARTS = {
    'Alcohol':      '[OX2H]',            # –OH
    'Carbonyl':     '[CX3]=O',           # C=O
    'Amine':        '[NX3;!H0]',         # –NH– or –NH2
    'Carboxylic':   'C(=O)[OX2H1]',       # –COOH
    'Aromatic ring':'a1aaaaa1',          # six-membered aromatic
}

# Pre‐compile tokenization regex (captures multi‐char tokens first)
TOKEN_REGEX = re.compile(
    r'\%\d{2}|'    # two‐digit ring closures like %10
    r'Br|Cl|'      # bromine, chlorine
    r'\[[^\]]+\]|' # bracket expressions [nH], etc.
    r'Si|Se|'      # silicon, selenium
    r'[B-IK-NOP-SU-Z]|'  # other uppercase atoms
    r'[bcnops]|'         # lowercase aromatic atoms
    r'\d|'               # ring closures 1–9
    r'[\=\#\-\+\:\\\/\.]'# bonds and other symbols
)

# -----------------------------
# 2. HELPERS
# -----------------------------
def tokenize_smiles(smiles):
    """Split SMILES into a list of atomic/bond tokens."""
    return TOKEN_REGEX.findall(smiles)

def mask_functional_group(mol, tokens):
    """
    Find one functional group match (at random) and mask all
    tokens corresponding to atoms in that substructure.
    Returns a new token list, or None if no FG found.
    """
    groups = []
    for smarts in FG_SMARTS.values():
        patt = Chem.MolFromSmarts(smarts)
        hit = mol.GetSubstructMatches(patt)
        if hit:
            groups.append(hit)

    if not groups:
        return None

    # choose one matched group
    atom_ids = random.choice(groups)
    # build a set of atom‐index‐to‐token‐positions
    # to map back to tokens, we regenerate the molecule with atom‐idx tags
    tagged = Chem.Mol(mol)
    for atom in tagged.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)
    tagged_smiles = Chem.MolToSmiles(tagged, canonical=True)

    # tokenize tagged SMILES and find map‐num tokens
    toks = tokenize_smiles(tagged_smiles)
    for i, tk in enumerate(toks):
        m = re.match(r'\[.*?:([0-9]+)\]', tk)  # e.g. "[C:5]"
        if m and int(m.group(1)) in atom_ids:
            toks[i] = MASK_TOKEN

    # remove map‐nums before returning
    joined = ''.join(toks)
    # strip out the :map inside brackets
    return re.sub(r'\[([^\[\]:]+):\d+\]', r'[\1]', joined)

def mask_random_tokens(tokens, prob=MASK_PROB):
    """Randomly mask individual tokens with MASK_TOKEN at given probability."""
    return [ (MASK_TOKEN if (random.random() < prob and tk not in '()[]=#+-\\/') else tk)
             for tk in tokens ]

# -----------------------------
# 3. MAIN
# -----------------------------
if __name__ == '__main__':
    df = pd.read_csv(INPUT_CSV)
    masked_list = []

    for smi in df[SMILES_COL]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            masked_list.append(None)
            continue

        # Try FG masking first
        tokens = tokenize_smiles(smi)
        fg_masked = mask_functional_group(mol, tokens)
        if fg_masked:
            masked_list.append(fg_masked)
        else:
            # fallback to random token masking
            rtoks = mask_random_tokens(tokens)
            masked_list.append(''.join(rtoks))

    df['masked_smiles'] = masked_list
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done — wrote masked SMILES to {OUTPUT_CSV}")#

# Do random token masking first