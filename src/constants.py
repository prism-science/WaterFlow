# constants.py
"""
Constants for edge types and other shared values.
"""

# Node feature dimensions
NODE_FEATURE_DIM = 16  # Default node scalar feature dimension

# Native widths of the cached embeddings produced by scripts/generate_*_embeddings.py.
ESM_EMBEDDING_DIM = 1536
SLAE_EMBEDDING_DIM = 128

# RBF (Radial Basis Function) parameters
NUM_RBF = 16  # Number of RBF basis functions
RBF_CUTOFF = 8.0  # Distance cutoff in Angstroms for RBF encoding

# Default radius for dynamic edge construction. Kept equal to RBF_CUTOFF on
# purpose: an edge longer than RBF_CUTOFF encodes to ~0 under the RBF, so
# building edges beyond it would feed the message passing near-zero features.
DEFAULT_EDGE_CUTOFF = RBF_CUTOFF

# van der Waals radius of oxygen, in Angstroms. Sets the clustering and
# non-maximum-suppression radius when merging predicted water candidates.
OXYGEN_VDW_RADIUS = 1.52

# Default minimum EDIAm score for a ground-truth water to be kept during
# training-data quality filtering. Waters scoring below this are dropped as
# unreliably placed. Shared default for the dataset filters and the CLIs.
DEFAULT_MIN_EDIA = 0.4

# Edge type tuples: (src_node_type, edge_name, dst_node_type)
EDGE_PP = ("protein", "pp", "protein")  # protein -> protein
EDGE_WW = ("water", "ww", "water")  # water -> water
EDGE_PW = ("protein", "pw", "water")  # protein -> water
EDGE_WP = ("water", "wp", "protein")  # water -> protein

# all edge types used in the model (future support: het atoms)
ALL_EDGE_TYPES = [EDGE_PW, EDGE_WW, EDGE_PP, EDGE_WP]


def get_active_edge_types(
    disable_ww: bool = False, disable_wp: bool = False
) -> list[tuple[str, str, str]]:
    """
    Return the active edge types for a model configuration.

    PW and PP are always active: PW carries protein context onto waters and PP
    is read from the geometry cache. WW and WP are ablatable.

    The returned order differs from ``ALL_EDGE_TYPES``. That is safe -- edge
    types key ``HeteroConv``'s parameters by name (``convs.<protein___pw___water>``),
    not by position, so ordering does not affect state-dict compatibility.

    Args:
        disable_ww: Drop water -> water edges.
        disable_wp: Drop water -> protein edges.

    Returns:
        List of (src_type, relation, dst_type) tuples.
    """
    etypes = [EDGE_PW, EDGE_PP]
    if not disable_ww:
        etypes.append(EDGE_WW)
    if not disable_wp:
        etypes.append(EDGE_WP)
    return etypes


# Standard 3-letter to 1-letter amino acid mapping
# Includes 20 canonical amino acids plus common non-standard residues
# Non-canonical residues not in this dict should be mapped to 'X'
THREE_TO_ONE = {
    # 20 canonical amino acids
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    # common non-standard amino acids
    "MSE": "M",  # Selenomethionine -> Methionine
    "SEC": "U",  # Selenocysteine
    "PYL": "O",  # Pyrrolysine
}

# Standard 1-to-3 letter mapping to feed sanitized residues back to ESM3.
# This acts as the inverse of THREE_TO_ONE, ensuring ESM3 recognizes
# the atoms and safely maps true unknowns to 'UNK'. (probably a more efficient way to do this I know)
ONE_TO_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "UNK",
    "U": "SEC",
    "O": "PYL",
}

# Element vocabulary for atom types in protein structures
ELEMENT_VOCAB = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "SE",
    "MG",
    "ZN",
    "CA",
    "FE",
    "NA",
    "K",
    "CL",
    "F",
    "BR",
]
ELEM_IDX = {e: i for i, e in enumerate(ELEMENT_VOCAB)}

# Index of oxygen in ELEMENT_VOCAB, used to build water (oxygen) node features.
OXYGEN_INDEX = ELEM_IDX["O"]
