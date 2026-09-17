"""
Dataset utilities for protein-water structure loading and preprocessing.

This module provides:
- PDB parsing with biotite and PyMOL for crystal contacts
- Per-water quality filtering (distance, EDIA, B-factor)
- Structure-level quality checks (CoM distance, clashes, chain interactions)
- ProteinWaterDataset: PyTorch Dataset returning HeteroData graphs
- get_dataloader: Convenience function for DataLoader creation
"""

from __future__ import annotations

import itertools
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

import biotite.structure as bts
import numpy as np
import pymol2
import torch
import torch.nn.functional as F
from biotite.structure.io.pdb import get_structure, PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure as get_structure_cif
from loguru import logger
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch_cluster import radius_graph
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm

from src.constants import (
    DEFAULT_EDGE_CUTOFF,
    DEFAULT_MIN_EDIA,
    EDGE_PP,
    ELEM_IDX,
    ELEMENT_VOCAB,
    NUM_RBF,
)
from src.utils import (
    compute_edge_features,
    normalize_ins_code,
    sanitize_res_names_for_esm,
)


# Per-directory record of the settings a geometry cache was built with.
FILTER_META_FILENAME = "_filter_meta.json"


def element_onehot(symbols: list[str]) -> Tensor:
    """One-hot encoding with 'other' bucket at end."""
    other_idx = len(ELEMENT_VOCAB)
    indices = torch.tensor(
        [ELEM_IDX.get(s.upper(), other_idx) for s in symbols], dtype=torch.long
    )
    return F.one_hot(indices, num_classes=other_idx + 1).float()


def _read_structure(path: str | Path, extra_fields=None) -> bts.AtomArray:
    """Read structure from PDB or CIF file, dispatching on extension."""
    path = Path(path)
    kw = dict(model=1, altloc="occupancy")
    if extra_fields:
        kw["extra_fields"] = extra_fields
    if path.suffix == ".cif":
        cif_file = CIFFile.read(path)
        return get_structure_cif(cif_file, **kw)
    else:
        pdb_file = PDBFile.read(path)
        return get_structure(pdb_file, **kw)


def parse_asu_with_biotite(
    path: str | Path,
) -> tuple[bts.AtomArray, bts.AtomArray, bts.AtomArray]:
    """
    Parse PDB or CIF file and extract protein, water, and ligand atoms.

    Args:
        path: Path to PDB or CIF file

    Returns:
        Tuple of (protein_atoms, water_atoms, ligand_atoms) as biotite AtomArrays.
        Hydrogen atoms are excluded. ligand_atoms contains every non-protein,
        non-water heavy atom: small-molecule ligands, ions, cofactors, AND
        non-amino-acid polymers such as nucleic acids (DNA/RNA). It is deliberately
        NOT restricted to HETATM records -- nucleic acids are written as ATOM
        records but are kept here as context (their surfaces, especially the
        phosphate backbone, order nearby water).

    Notes:
        - model=1: Uses first model in PDB (standard for X-ray structures)
        - altloc="occupancy": Selects highest-occupancy alternate conformation
        - Uses filter_amino_acids (not filter_canonical_amino_acids) to include
          modified residues like MSE, SEC that external encoders may handle
        - b_factor is always read so the caller can compute normalized B-factors
          without a second file read.
    """
    atoms = _read_structure(path, extra_fields=["b_factor"])

    atoms = atoms[atoms.element != "H"]

    protein_mask = bts.filter_amino_acids(atoms)
    water_mask = (atoms.res_name == "HOH") | (atoms.res_name == "WAT")

    # "ligand" here is broad: every non-protein, non-water heavy atom.
    # includes small-molecule ligands, ions, cofactors and even nucleic acids
    ligand_mask = ~protein_mask & ~water_mask

    protein_atoms = atoms[protein_mask]
    water_atoms = atoms[water_mask]
    ligand_atoms = atoms[ligand_mask]

    return protein_atoms, water_atoms, ligand_atoms


def crystal_symmetry_check(struc_path: str) -> str | None:
    """Say why a file cannot produce symmetry mates, or return None if it can.

    symexp silently yields nothing when the file has no symmetry record, and
    yields stacked translational copies of the whole structure when the cell
    is a placeholder (predicted and NMR files often carry CRYST1 1 1 1 P 1).
    A real crystal cell has at least ~15 cubic Angstroms per atom, a
    placeholder ~1 total, so a floor of 4 per atom separates them with wide
    margin.
    """
    with pymol2.PyMOL() as pm:
        cmd = pm.cmd
        cmd.feedback("disable", "all", "everything")
        obj = "struct"
        cmd.load(struc_path, obj)
        symmetry = cmd.get_symmetry(obj)
        if symmetry is None:
            return "no crystal symmetry record"
        if symmetry[0] * symmetry[1] * symmetry[2] < 4.0 * cmd.count_atoms(obj):
            return (
                f"placeholder unit cell {symmetry[0]:g} x {symmetry[1]:g} x "
                f"{symmetry[2]:g} too small to hold the structure"
            )
        return None


def get_crystal_contacts_pymol(
    struc_path: str,
    cutoff: float = 5.0,
    include_ligands: bool = False,
) -> dict[str, np.ndarray | list]:
    """
    Extract ASU and symmetry-mate atoms within crystal contact distance.

    PyMOL's symexp generates the mates; `byres` then keeps whole residues and
    whole ligand entities with any atom within `cutoff` of the ASU. Protein and
    ligand mates come back under separate keys, classified by PyMOL itself
    (`polymer.protein` vs the non-protein, non-solvent remainder), so nothing
    downstream needs residue-name heuristics.

    Mate waters are never selected: a mate water is a symmetry image of an ASU
    water, which is a prediction target, so keeping it as context is a label
    leak. Protein and ligand contact surfaces are genuine context.

    Args:
        struc_path: Structure file (PDB/CIF) carrying crystal symmetry.
        cutoff: Interface distance cutoff in Angstroms.
        include_ligands: Also collect whole ligand/ion/cofactor/nucleic-acid mate
            entities (still never waters). Off by default: protein mates alone.

    Returns:
        Dict with keys:
            - 'asu_coords': (N_asu, 3) ASU atom coordinates
            - 'asu_atoms': List of PyMOL atom objects for the ASU
            - 'mate_coords': (N_mate, 3) whole protein-mate residues
            - 'mate_atoms': List of PyMOL atom objects for protein mates
            - 'mate_ligand_coords': (M, 3) whole ligand-mate entities, empty
              unless include_ligands
            - 'mate_ligand_atoms': List of PyMOL atom objects for ligand mates

    Raises:
        ValueError: If the file has no symmetry record or only a placeholder
            unit cell, as predicted and NMR structures do.
    """
    error = crystal_symmetry_check(struc_path)
    if error is not None:
        raise ValueError(
            f"{struc_path}: {error}; cannot build symmetry mates. Run "
            "this structure with a model trained without mates, e.g. the "
            "released mates_off checkpoints."
        )

    with pymol2.PyMOL() as pm:
        cmd = pm.cmd
        cmd.reinitialize()
        cmd.feedback("disable", "all", "everything")
        obj = "struct"
        cmd.load(struc_path, obj)
        cmd.symexp("sym", obj, obj, cutoff)

        def _coords(selection: str) -> np.ndarray:
            coords = cmd.get_coords(selection, state=1)
            return coords if coords is not None else np.zeros((0, 3), dtype=float)

        # Whole protein-mate residues with any atom within cutoff of the ASU.
        # `not hydro` last: byres would otherwise re-add the residue's hydrogens,
        # and mates must stay heavy-atom-only like the ASU.
        cmd.select(
            "iface_prot",
            f"(byres ((sym* and polymer.protein) within {cutoff} of {obj})) "
            f"and not hydro",
        )
        mate_coords = _coords("iface_prot")
        mate_atoms = cmd.get_model("iface_prot", state=1).atom

        # Whole ligand-mate entities (non-protein, non-water het atoms: ligands,
        # ions, cofactors, nucleic acids).
        if include_ligands:
            cmd.select(
                "iface_lig",
                f"(byres ((sym* and (not polymer.protein) and (not solvent)) "
                f"within {cutoff} of {obj})) and not hydro",
            )
            mate_ligand_coords = _coords("iface_lig")
            mate_ligand_atoms = cmd.get_model("iface_lig", state=1).atom
        else:
            mate_ligand_coords = np.zeros((0, 3), dtype=float)
            mate_ligand_atoms = []

        return {
            "asu_coords": _coords(obj),
            "asu_atoms": cmd.get_model(obj, state=1).atom,
            "mate_coords": mate_coords,
            "mate_atoms": mate_atoms,
            "mate_ligand_coords": mate_ligand_coords,
            "mate_ligand_atoms": mate_ligand_atoms,
        }


def match_atoms_to_coords(
    atoms: bts.AtomArray, target_coords: np.ndarray, tolerance: float = 0.01
) -> list[int]:
    """
    Match biotite atoms to coordinates from a second parse of the same structure.

    Reconciles biotite against PyMOL. The two disagree on count by design: PyMOL
    keeps every altloc conformer while biotite takes the highest-occupancy one,
    so PyMOL's atom set is a superset. Every biotite atom should still be found
    in it; the caller drops any that are not.

    Args:
        atoms: Biotite AtomArray with coord attribute
        target_coords: (N, 3) coordinates to match against
        tolerance: Maximum distance in Angstroms for a valid match

    Returns:
        Index into atoms for each target coordinate whose nearest atom lies
        within tolerance. May repeat an index if two targets share an atom.
    """
    if target_coords.shape[0] == 0 or len(atoms) == 0:
        return []

    tree = cKDTree(atoms.coord)
    dists, nearest = tree.query(target_coords, k=1, distance_upper_bound=tolerance)
    within = np.isfinite(dists) & (nearest < len(atoms))
    matched = nearest[within].tolist()

    # A wholesale miss means the parses disagree (frame, cell), not that the
    # atoms are bad. Warn, or the caller's drop looks like clean data.
    if len(set(matched)) < len(atoms) / 2:
        logger.warning(
            f"Only {len(set(matched))}/{len(atoms)} atoms matched within "
            f"{tolerance}A; parses may disagree. Unmatched atoms are dropped."
        )
    return matched


def dedup_mate_atoms(
    mate_coords: np.ndarray,
    mate_atoms: list,
    reference_coords: np.ndarray,
    tol: float = 0.3,
) -> tuple[np.ndarray, list]:
    """
    Drop mate atoms coincident with a reference atom or an already-kept mate atom.

    Crystal symmetry creates coincidences: an atom on a rotation or screw axis
    maps onto itself, and one residue can be reached through two operators. Left
    alone each becomes an independent node, giving duplicates joined by ~0 A
    edges -- and, for a target water on a special position, a label leak.

    Per atom, unlike `dedup_mate_ligands_by_residue`: a special position is an
    atom-level accident, so only the coincident atom is dropped. The self sweep
    also catches mates coincident with each other, which the reference tree cannot.

    Args:
        mate_coords: (N, 3) mate atom coordinates, uncentered.
        mate_atoms: Parallel list of mate atom objects, kept in lockstep.
        reference_coords: (M, 3) uncentered ASU coordinates.
        tol: Coincidence radius in Angstroms.

    Returns:
        (kept_coords, kept_atoms). The first atom of a coincident group is the
        one kept, so the result depends on input order.
    """
    n = mate_coords.shape[0]
    if n == 0:
        return mate_coords, mate_atoms

    # An empty reference tree answers inf, so no guard is needed here.
    drop = cKDTree(reference_coords).query(mate_coords, k=1)[0] < tol

    # Self-dedup is a first-win sweep. One tree answers every lookup, so the
    # sweep only walks each atom's coincident neighbors.
    neighbors = cKDTree(mate_coords).query_ball_point(mate_coords, r=tol)
    kept = np.zeros(n, dtype=bool)
    for i in range(n):
        if drop[i]:
            continue
        earlier = [j for j in neighbors[i] if j < i and kept[j]]
        # query_ball_point includes r, this sweep is strict.
        dists = np.linalg.norm(mate_coords[earlier] - mate_coords[i], axis=1)
        kept[i] = not (dists < tol).any()

    keep_idx = np.flatnonzero(kept)
    return mate_coords[keep_idx], [mate_atoms[i] for i in keep_idx]


def dedup_mate_ligands_by_residue(
    lig_coords: np.ndarray,
    lig_atoms: list,
    reference_coords: np.ndarray,
    tol: float = 0.3,
    image_frac: float = 0.5,
) -> tuple[np.ndarray, list]:
    """
    Drop whole mate-ligand entities that are symmetry images of ASU atoms.

    Unlike `dedup_mate_atoms`, which works per atom, this works per entity so a
    ligand is never fragmented: it goes only when the whole ligand is a redundant
    copy. Genuine neighbor-cell ligands are kept whole.

    Args:
        lig_coords: (M, 3) mate-ligand atom coordinates, uncentered.
        lig_atoms: Parallel list of mate-ligand atom objects.
        reference_coords: Uncentered ASU coordinates.
        tol: Coincidence radius in Angstroms.
        image_frac: Drop a ligand when more than this fraction of its atoms are
            coincident with the reference.

    Returns:
        (kept_coords, kept_atoms) with whole symmetry-image ligands removed.
    """
    if len(lig_atoms) == 0:
        return lig_coords, lig_atoms

    ref_tree = cKDTree(reference_coords)
    # Group atom indices by ligand entity. atom.model is the symmetry-object id, so
    # two neighbour-cell images of one ligand stay separate entities: without it
    # they collide into one group and a single image_frac is averaged over both,
    # which can dilute a genuine ASU-image copy below the drop threshold.
    groups = {}
    for i, atom in enumerate(lig_atoms):
        key = (
            getattr(atom, "model", ""),
            atom.chain,
            atom.resi,
            getattr(atom, "segi", ""),
        )
        groups.setdefault(key, []).append(i)

    keep_idx: list[int] = []
    for idxs in groups.values():
        if np.mean(ref_tree.query(lig_coords[idxs], k=1)[0] < tol) <= image_frac:
            keep_idx.extend(idxs)  # genuine neighbor ligand: keep whole
    keep_idx.sort()

    return lig_coords[keep_idx], [lig_atoms[i] for i in keep_idx]


def _parse_pdb_resi(resi) -> tuple[int, str] | None:
    """
    Parse a PyMOL residue identifier, which may carry an insertion code.

    Args:
        resi: Residue id as PyMOL exposes it, e.g. "52", "-3", "52A".

    Returns:
        (res_id, ins_code), or None when there is no integer part, which the
        caller scores with a zero embedding rather than crashing on.
    """
    match = re.match(r"^\s*(-?\d+)\s*([A-Za-z]?)\s*$", str(resi))
    if match is None:
        return None
    return int(match.group(1)), match.group(2).strip()


def _make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Convert directed edges to undirected by adding reverse edges.

    Args:
        edge_index: (2, E) directed edge index tensor

    Returns:
        (2, E') undirected edge index with reverse edges added and duplicates removed
    """
    if edge_index.numel() == 0:
        return edge_index
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # add reverse edges
    ei = torch.unique(ei.T, dim=0).T  # drop duplicates
    return ei


def _pad_atom_embeddings_for_mates(
    asu_embedding: torch.Tensor,
    total_num_atoms: int,
) -> torch.Tensor:
    """
    Pad ASU-only atom embeddings with zeros for symmetry mate atoms.

    Args:
        asu_embedding: (N_asu, embed_dim) embeddings for ASU atoms only
        total_num_atoms: Total number of atoms including symmetry mates

    Returns:
        (total_num_atoms, embed_dim) padded embeddings with zeros for mate atoms
    """
    if total_num_atoms <= asu_embedding.size(0):
        return asu_embedding
    pad = asu_embedding.new_zeros(
        total_num_atoms - asu_embedding.size(0), asu_embedding.size(1)
    )
    return torch.cat([asu_embedding, pad], dim=0)


def _load_torch_cache(path: Path, cache_load_mmap: bool = True) -> dict:
    """Load a torch cache file, using mmap when supported by the file/runtime."""
    if not cache_load_mmap:
        return torch.load(path, weights_only=False)

    try:
        return torch.load(path, weights_only=False, mmap=True)
    except (TypeError, ValueError, RuntimeError, OSError) as exc:
        logger.debug(f"mmap torch.load failed for {path}; falling back: {exc}")
        return torch.load(path, weights_only=False)


def load_slae_embedding(
    embedding_dir: Path,
    cache_key: str,
    num_asu_protein: int,
    total_num_atoms: int,
    cache_load_mmap: bool = True,
) -> torch.Tensor:
    """
    Load SLAE atom-level embeddings from cache.

    This is a standalone function to allow reuse outside dataset context.

    Args:
        embedding_dir: Directory containing cached embedding files
        cache_key: Identifier for the cached embedding file
        num_asu_protein: Expected number of ASU protein atoms
        total_num_atoms: Total protein atoms including symmetry mates
        cache_load_mmap: Use mmap-backed torch.load when supported

    Returns:
        (total_num_atoms, slae_dim) tensor with zeros padded for mate atoms

    Raises:
        FileNotFoundError: If SLAE cache file doesn't exist
        ValueError: If atom count doesn't match expected ASU count
    """
    slae_cache_path = embedding_dir / f"{cache_key}.pt"
    if not slae_cache_path.exists():
        raise FileNotFoundError(
            f"SLAE cache file not found: {slae_cache_path}. "
            "Generate embeddings with scripts/generate_slae_embeddings.py."
        )
    slae_cached = _load_torch_cache(slae_cache_path, cache_load_mmap=cache_load_mmap)
    if "node_embeddings" not in slae_cached:
        raise KeyError(f"Missing 'node_embeddings' in SLAE cache: {slae_cache_path}")
    slae_emb = slae_cached["node_embeddings"]
    if slae_emb.size(0) != num_asu_protein:
        raise ValueError(
            f"SLAE embedding atom count mismatch for {cache_key}: "
            f"expected {num_asu_protein}, got {slae_emb.size(0)}"
        )
    return _pad_atom_embeddings_for_mates(slae_emb, total_num_atoms)


def load_esm_embedding(
    embedding_dir: Path,
    cache_key: str,
    num_protein_residues: int,
    cache_load_mmap: bool = True,
) -> torch.Tensor:
    """
    Load ESM residue-level embeddings from cache.

    This is a standalone function to allow reuse outside dataset context.
    Returns raw residue embeddings; broadcasting to atom level is done separately.

    Args:
        embedding_dir: Directory containing cached embedding files
        cache_key: Identifier for the cached embedding file
        num_protein_residues: Expected number of unique residues
        cache_load_mmap: Use mmap-backed torch.load when supported

    Returns:
        (num_protein_residues, esm_dim) tensor of residue embeddings

    Raises:
        FileNotFoundError: If ESM cache file doesn't exist
        ValueError: If residue count doesn't match expected count
    """
    esm_cache_path = embedding_dir / f"{cache_key}.pt"
    if not esm_cache_path.exists():
        raise FileNotFoundError(
            f"ESM cache file not found: {esm_cache_path}. "
            "Generate embeddings with scripts/generate_esm_embeddings.py."
        )
    esm_cached = _load_torch_cache(esm_cache_path, cache_load_mmap=cache_load_mmap)
    if "residue_embeddings" not in esm_cached:
        raise KeyError(f"Missing 'residue_embeddings' in ESM cache: {esm_cache_path}")
    residue_embeddings = esm_cached["residue_embeddings"]
    if residue_embeddings.size(0) != num_protein_residues:
        raise ValueError(
            f"ESM residue count mismatch for {cache_key}: "
            f"expected {num_protein_residues}, got {residue_embeddings.size(0)}"
        )
    return residue_embeddings


def check_com_distance(
    protein_coords: torch.Tensor,
    water_coords: torch.Tensor,
    max_com_dist: float = 25.0,
) -> tuple[bool, str]:
    """
    Check if protein and water centers of mass are within acceptable distance.

    Large CoM differences indicate atoms are in different frames of reference.

    Args:
        protein_coords: (N, 3) protein atom coordinates
        water_coords: (M, 3) water atom coordinates
        max_com_dist: Maximum allowed distance between CoMs (Angstroms)

    Returns:
        (is_valid, reason) tuple
    """
    if water_coords.size(0) == 0:
        return True, ""

    protein_com = protein_coords.mean(dim=0)
    water_com = water_coords.mean(dim=0)
    com_dist = torch.linalg.norm(protein_com - water_com).item()

    if com_dist > max_com_dist:
        return False, f"CoM distance {com_dist:.1f}A exceeds threshold {max_com_dist}A"
    return True, ""


def check_water_clashes(
    protein_coords: torch.Tensor,
    water_coords: torch.Tensor,
    clash_dist: float = 2.0,
    max_clash_fraction: float = 0.05,
) -> tuple[bool, str]:
    """
    Check if too many waters clash with the protein surface (within a threshold).

    Waters within clash_dist of any protein atom are considered clashing.

    Args:
        protein_coords: (N, 3) protein atom coordinates
        water_coords: (M, 3) water atom coordinates
        clash_dist: Distance threshold for clash detection (Angstroms)
        max_clash_fraction: Maximum allowed fraction of clashing waters (0-1)

    Returns:
        (is_valid, reason) tuple
    """
    if water_coords.size(0) == 0:
        return True, ""

    # compute pairwise distances: (M, N)
    dists = torch.cdist(water_coords, protein_coords)
    min_dists = dists.min(dim=1).values  # closest protein atom to each water

    n_clashing = (min_dists < clash_dist).sum().item()
    clash_fraction = n_clashing / water_coords.size(0)

    if clash_fraction > max_clash_fraction:
        return False, (
            f"Water clash fraction {clash_fraction:.1%} ({n_clashing}/{water_coords.size(0)}) "
            f"exceeds threshold {max_clash_fraction:.0%}"
        )
    return True, ""


def check_chain_interactions(
    protein_atoms: bts.AtomArray,
    interface_dist_threshold: float = 4.0,
) -> tuple[bool, str, str]:
    """
    Check if multi-chain proteins have interacting chains (PPI) vs ASU copies.

    For proteins with >=2 chains, computes minimum inter-chain distance.
    If min distance > threshold, chains are likely ASU copies, not a true PPI.

    Args:
        protein_atoms: biotite AtomArray with chain_id and coord attributes
        interface_dist_threshold: Chains must be within this distance to be
            considered interacting (Angstroms)

    Returns:
        (is_valid, reason, interaction_status) tuple where interaction_status
        is one of: "Single Chain", "Interacting", "Non-Interacting (ASU Copies)"
    """
    chain_ids = np.unique(protein_atoms.chain_id)
    num_chains = len(chain_ids)

    if num_chains < 2:
        return True, "", "Single Chain"

    chain_coords = {
        cid: torch.tensor(
            protein_atoms[protein_atoms.chain_id == cid].coord, dtype=torch.float32
        )
        for cid in chain_ids
    }

    min_interface_dist = float("inf")
    for chain_a, chain_b in itertools.combinations(chain_ids, 2):
        coords_a = chain_coords[chain_a]
        coords_b = chain_coords[chain_b]
        min_d = torch.cdist(coords_a, coords_b).min().item()
        if min_d < min_interface_dist:
            min_interface_dist = min_d

    if min_interface_dist > interface_dist_threshold:
        return (
            False,
            f"Multi-chain ({num_chains} chains) min interface distance {min_interface_dist:.1f}A "
            f"> {interface_dist_threshold}A (likely ASU copies, not PPI)",
            "Non-Interacting (ASU Copies)",
        )

    return True, "", "Interacting"


def check_water_residue_ratio(
    num_waters: int,
    num_residues: int,
    min_ratio: float = 0.8,
) -> tuple[bool, str]:
    """
    Check if water/residue ratio meets minimum threshold.

    Structures with too few waters relative to protein size may be
    poorly resolved or have incomplete solvent modeling.

    Args:
        num_waters: Number of water molecules
        num_residues: Number of protein residues
        min_ratio: Minimum required waters/residues ratio

    Returns:
        (is_valid, reason) tuple
    """
    if num_residues == 0:
        return False, "No residues found"

    ratio = num_waters / num_residues

    if ratio < min_ratio:
        return False, (
            f"Water/residue ratio {ratio:.2f} ({num_waters}/{num_residues}) "
            f"below threshold {min_ratio}"
        )
    return True, ""


def load_edia_for_pdb(
    json_path: str | Path,
) -> dict[tuple[str, int, str], float] | None:
    """
    Load EDIA scores for water molecules from a JSON file.

    Args:
        json_path: Path to JSON file containing EDIA scores for the structure

    Returns:
        Dictionary mapping (chain_id, res_id, ins_code) -> EDIA score for waters,
        or None if file not found or error
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        edia_lookup = {}
        for entry in data:
            # Filter for water molecules only
            if entry.get("compID") in ["HOH", "WAT"]:
                # The identifying information is nested inside the "pdb" key in the JSON
                pdb_info = entry.get("pdb", {})

                chain_id = str(pdb_info.get("strandID", ""))
                res_id = int(pdb_info.get("seqNum", 0))

                # Extract and normalize insertion code, defaulting to an empty string
                raw_ins_code = pdb_info.get("insCode", "")
                ins_code = normalize_ins_code(raw_ins_code) if raw_ins_code else ""

                # Build the lookup key and extract the EDIAm score
                key = (chain_id, res_id, ins_code)
                edia_lookup[key] = float(entry.get("EDIAm", 0.0))

        if not edia_lookup:
            return {}

        return edia_lookup

    except Exception as e:
        logger.warning(f"Warning: Could not load EDIA JSON data for {json_path}: {e}")
        return None


def compute_normalized_bfactors(
    struc_path: str,
) -> tuple[dict[tuple[str, int, str], float] | None, np.ndarray | None]:
    """
    Extract and normalize B-factors for water molecules.

    B-factors are z-score normalized using statistics from water atoms only
    in the selected structure.

    Args:
        struc_path: Path to structure file (PDB/CIF)

    Returns:
        Tuple of:
        - Dictionary mapping (chain_id, res_id, ins_code) -> normalized B-factor for waters
        - Raw B-factor array for waters (for caching if needed)
        Returns (None, None) on error
    """
    try:
        atoms = _read_structure(struc_path, extra_fields=["b_factor"])

        # filter for water molecules
        water_mask = (atoms.res_name == "HOH") | (atoms.res_name == "WAT")
        water_atoms = atoms[water_mask]

        return _compute_normalized_bfactors_from_atoms(water_atoms)

    except Exception as e:
        logger.warning(f"Warning: Could not extract B-factors from {struc_path}: {e}")
        return None, None


def _compute_normalized_bfactors_from_atoms(
    water_atoms: bts.AtomArray,
) -> tuple[dict[tuple[str, int, str], float] | None, np.ndarray | None]:
    """Compute normalized B-factors from an already-parsed water AtomArray."""
    try:
        if not water_atoms:
            return None, None

        water_mean = np.mean(water_atoms.b_factor)
        water_std = np.std(water_atoms.b_factor)

        bfactor_lookup = {}
        for i in range(len(water_atoms)):
            chain_id = str(water_atoms.chain_id[i])
            res_id = int(water_atoms.res_id[i])
            ins_code = normalize_ins_code(water_atoms.ins_code[i])
            key = (chain_id, res_id, ins_code)
            if key not in bfactor_lookup:
                raw_bfactor = water_atoms.b_factor[i]
                normalized = (
                    (raw_bfactor - water_mean) / max(water_std, 1e-3)
                    if water_std > 0
                    else 0.0
                )
                bfactor_lookup[key] = normalized

        return bfactor_lookup, water_atoms.b_factor

    except Exception as e:
        logger.warning(f"Warning: Could not compute B-factors from atoms: {e}")
        return None, None


def apply_threshold_filter(
    water_keys: list[tuple],
    lookup: dict[tuple, float],
    threshold: float,
    fail_if_below: bool,
) -> np.ndarray:
    """
    Apply a threshold filter using a lookup dictionary.

    Args:
        water_keys: List of per-water residue keys
        lookup: Dict mapping residue key -> value
        threshold: Threshold value for comparison
        fail_if_below: If True, fail when value < threshold (e.g., EDIA).
                       If False, fail when value > threshold (e.g., B-factor).

    Returns:
        Boolean mask where True indicates the water FAILS the filter.
        Waters missing from lookup get np.nan and pass the filter (conservative).
    """
    values = np.array([lookup.get(key, np.nan) for key in water_keys])
    if fail_if_below:
        return values < threshold
    return values > threshold


def filter_waters_by_quality(
    water_coords: np.ndarray,
    water_keys: list[tuple],
    protein_coords: np.ndarray | None,
    edia_lookup: dict[tuple, float] | None,
    bfactor_lookup: dict[tuple, float] | None,
    max_protein_dist: float = 6.0,
    min_edia: float = DEFAULT_MIN_EDIA,
    max_bfactor_zscore: float = 1.5,
    cache_key: str | None = None,
) -> np.ndarray:
    """
    Filter water atoms based on quality criteria.

    Waters are removed if they fail ANY of the enabled criteria:
    1. Distance from protein surface > max_protein_dist (if protein_coords provided)
    2. EDIA score < min_edia (if edia_lookup provided)
    3. Normalized B-factor > max_bfactor_zscore (if bfactor_lookup provided)

    Args:
        water_coords: (N, 3) array of water coordinates
        water_keys: List of per-water residue keys
        protein_coords: (M, 3) array of protein coordinates, or None to skip distance filtering
        edia_lookup: Dict mapping residue key -> EDIA score, or None to skip EDIA filtering
        bfactor_lookup: Dict mapping residue key -> normalized B-factor, or None to skip B-factor filtering
        max_protein_dist: Maximum allowed distance to protein surface
        min_edia: Minimum allowed EDIA score
        max_bfactor_zscore: Maximum allowed B-factor z-score
        cache_key: Optional identifier for logging (e.g., PDB ID)

    Returns:
        np.ndarray: Boolean mask of waters to keep (True = keep, False = remove)
    """
    n_waters = len(water_keys)

    if n_waters == 0:
        return np.array([], dtype=bool)

    stats = {
        "total": n_waters,
        "removed_distance": 0,
        "removed_edia": 0,
        "removed_bfactor": 0,
    }

    # distance filtering using scipy.spatial.distance.cdist
    dist_fail = np.zeros(n_waters, dtype=bool)
    if protein_coords is not None and len(protein_coords):
        dist_matrix = cdist(water_coords, protein_coords)
        min_dists = dist_matrix.min(axis=1)
        dist_fail = min_dists > max_protein_dist
        stats["removed_distance"] = int(dist_fail.sum())

    # lookup-based filters: (lookup, threshold, fail_if_below, stat_key)
    lookup_filters = [
        (edia_lookup, min_edia, True, "edia"),
        (bfactor_lookup, max_bfactor_zscore, False, "bfactor"),
    ]

    lookup_fail = np.zeros(n_waters, dtype=bool)
    for lookup, threshold, fail_if_below, name in lookup_filters:
        if lookup is not None:
            fail_mask = apply_threshold_filter(
                water_keys, lookup, threshold, fail_if_below
            )
            stats[f"removed_{name}"] = int(fail_mask.sum())
            lookup_fail |= fail_mask

    # combine all failure masks - water is kept only if it passes all enabled filters
    keep_mask = ~(dist_fail | lookup_fail)
    stats["kept"] = int(keep_mask.sum())

    # log filtering statistics
    if cache_key is not None and stats["total"] > 0:
        removed = stats["total"] - stats["kept"]
        if removed > 0:
            logger.info(
                f"  {cache_key}: Filtered {removed}/{stats['total']} waters "
                f"(dist:{stats['removed_distance']}, "
                f"edia:{stats['removed_edia']}, "
                f"bfactor:{stats['removed_bfactor']})"
            )

    return keep_mask


class ProteinWaterDataset(Dataset):
    """
    Dataset for predicting water positions in protein crystal structures.

    Returns HeteroData with:
    - 'protein' node type: ASU protein atoms + optionally symmetry mates
    - 'water' node type: water molecules
    - ('protein', 'pp', 'protein') edges
    """

    def __init__(
        self,
        pdb_list_file: str,
        processed_dir: str,
        base_pdb_dir: str,
        encoder_type: str = "gvp",
        cutoff: float = DEFAULT_EDGE_CUTOFF,
        max_neighbors: int = 256,
        include_mates: bool = True,
        include_ligands: bool = True,
        geometry_cache_name: str = "geometry",
        preprocess: bool = True,
        duplicate_single_sample: int = 1,
        max_com_dist: float = 25.0,
        max_clash_fraction: float = 0.05,
        clash_dist: float = 2.0,
        interface_dist_threshold: float = 4.0,
        min_water_residue_ratio: float = 0.1,
        max_protein_dist: float = 5.0,
        min_edia: float = DEFAULT_MIN_EDIA,
        max_bfactor_zscore: float = 2.0,
        filter_by_distance: bool = True,
        filter_by_edia: bool = True,
        filter_by_bfactor: bool = True,
        sample_cache_size: int = 0,
        cache_load_mmap: bool = False,
    ):
        """
        Args:
            pdb_list_file: Text file with lines like "<pdb_id>_final"
            processed_dir: Cache root directory. Geometry caches are stored in
                           {processed_dir}/{geometry_cache_name}[_mates][_noligands]
                           and embedding caches in {processed_dir}/{encoder_name}.
            base_pdb_dir: Base directory containing PDB subdirectories
            encoder_type: Encoder used downstream ('gvp', 'slae', or 'esm').
                          Embeddings are loaded only for the selected type.
            cutoff: Distance cutoff for PP edges and crystal contacts (Angstroms)
            max_neighbors: Maximum neighbors per node for radius graph construction.
            include_mates: If True, include symmetry mate atoms as protein nodes
            include_ligands: If True (default), include every non-protein,
                             non-water heavy atom (small-molecule ligands, ions,
                             cofactors, and nucleic acids) as protein-type nodes.
                             They are appended after protein (and mate) atoms with a
                             boolean is_ligand mask and residue_index = -1.
            geometry_cache_name: Base name for geometry cache directory. Flags that
                                 change the cached node set are appended to it:
                                 "_mates" when include_mates=True, "_noligands" when
                                 include_ligands=False. Default is "geometry", yielding
                                 "geometry_mates/" for the default config or e.g.
                                 "geometry_mates_noligands/" with ligands excluded.
            preprocess: If True, run preprocessing on missing cached files
            duplicate_single_sample: If dataset has 1 sample, duplicate it this many times
            Quality checks (always active):
            max_com_dist: Max allowed distance between protein and water CoM (Angstroms).
                          Structures exceeding this are filtered (different reference frames).
            max_clash_fraction: Max fraction of waters allowed within clash_dist of protein.
                                Structures exceeding this are filtered.
            clash_dist: Distance threshold for water-protein clashes (Angstroms).
            interface_dist_threshold: For multi-chain proteins, min inter-chain distance
                                      must be <= this to be considered interacting.
                                      Structures with larger distances are filtered (ASU copies).
            min_water_residue_ratio: Minimum ratio of waters/residues required.
                                     Structures below this are filtered (poor solvent modeling).

            Per-water filtering (toggleable):
            max_protein_dist: Remove waters farther than this from nearest protein atom (Angstroms).
            min_edia: Remove waters with EDIA score below this threshold.
            max_bfactor_zscore: Remove waters with normalized B-factor (z-score) above this.
            filter_by_distance: Enable/disable distance-from-protein filtering.
            filter_by_edia: Enable/disable EDIA score filtering.
            filter_by_bfactor: Enable/disable B-factor z-score filtering.
                              If a per-water filter is disabled, its threshold is ignored.
            sample_cache_size: Number of fully built HeteroData samples to keep in a
                               per-process LRU cache. 0 disables sample caching.
            cache_load_mmap: Use mmap-backed torch.load for cache files when supported.
        """

        if sample_cache_size < 0:
            raise ValueError("sample_cache_size must be >= 0")
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be >= 1")

        self.cache_dir = Path(processed_dir)
        # Directory-based separation: geometry/ vs geometry_mates/. Both flags change
        # the cached node set, so both are encoded in the directory name -- otherwise
        # toggling one would silently reuse geometry built under the other setting.
        cache_suffix = "_mates" if include_mates else ""
        if not include_ligands:
            cache_suffix += "_noligands"
        self.geometry_dir = self.cache_dir / f"{geometry_cache_name}{cache_suffix}"
        self.base_pdb_dir = Path(base_pdb_dir)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.encoder_type = encoder_type
        if self.encoder_type in ("slae", "esm"):
            self.embedding_dir = self.cache_dir / self.encoder_type
        else:
            self.embedding_dir = None
        self.include_mates = include_mates
        self.include_ligands = include_ligands
        self.duplicate_single_sample = duplicate_single_sample

        self.max_com_dist = max_com_dist
        self.max_clash_fraction = max_clash_fraction
        self.clash_dist = clash_dist
        self.interface_dist_threshold = interface_dist_threshold
        self.min_water_residue_ratio = min_water_residue_ratio

        self.max_protein_dist = max_protein_dist
        self.min_edia = min_edia
        self.max_bfactor_zscore = max_bfactor_zscore
        self.filter_by_distance = filter_by_distance
        self.filter_by_edia = filter_by_edia
        self.filter_by_bfactor = filter_by_bfactor
        self.sample_cache_size = int(sample_cache_size)
        self.cache_load_mmap = bool(cache_load_mmap)
        self._sample_cache: OrderedDict[tuple[int, str], HeteroData] = OrderedDict()

        if self.encoder_type not in {"gvp", "slae", "esm"}:
            raise ValueError(
                f"Unsupported encoder_type '{self.encoder_type}'. "
                "Expected one of: gvp, slae, esm"
            )

        self.entries = self._parse_pdb_list(pdb_list_file)

        self._sync_filter_meta(write=preprocess)

        if preprocess:
            self._preprocess_all()

        # if single sample and duplication requested, set effective length [this is for experiments to check if the model can memorize a sample]
        if len(self.entries) == 1 and duplicate_single_sample > 1:
            self._effective_length = duplicate_single_sample
            logger.info(
                f"Single sample detected. Duplicating {duplicate_single_sample}x "
            )
        else:
            self._effective_length = len(self.entries)

    def _parse_pdb_list(self, pdb_list_file: str) -> list[dict]:
        """
        Parse PDB list file and construct entries with paths.

        Expected format:
        <pdb_id>_final  (e.g., "6eey_final")

        Resolves path in {base_pdb_dir}/{pdb_id}/, preferring
        {pdb_id}_final.cif when present, otherwise falling back to
        {pdb_id}_final.pdb.
        """
        entries = []
        logger.info(f"Parsing PDB list: {pdb_list_file}")
        pdb_ids = []
        with open(pdb_list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not line.endswith("_final"):
                    logger.warning(f"Warning: Unexpected format: {line}")
                    continue
                pdb_id = line.removesuffix("_final")
                if not pdb_id:
                    logger.warning(f"Warning: Unexpected format: {line}")
                    continue
                pdb_ids.append((pdb_id, line))

        logger.info(
            f"Read {len(pdb_ids)} IDs, resolving file paths for requested entries..."
        )

        for pdb_id, cache_key in pdb_ids:
            subdir = self.base_pdb_dir / pdb_id
            cif_path = subdir / f"{pdb_id}_final.cif"
            struc_path = (
                cif_path if cif_path.is_file() else subdir / f"{pdb_id}_final.pdb"
            )

            # Cache key is just the base key - directory separation handles mates
            entries.append(
                {
                    "pdb_id": pdb_id,
                    "struc_path": struc_path,
                    "cache_key": cache_key,
                    "embedding_key": cache_key,  # Same as cache_key for embedding lookup
                }
            )

        logger.info(f"Loaded {len(entries)} entries from {pdb_list_file}")
        return entries

    def _sync_filter_meta(self, write: bool) -> None:
        """
        Refuse to read or extend a cache built under different settings.

        Filtering happens before the cache is written, so these are properties of
        the directory rather than of the run reading it -- and the .pt files
        record none of them. Writing entries under different settings would leave
        one directory holding two populations no later reader can tell apart.

        Args:
            write: Create when it is absent. Only runs that may add
                entries (preprocess=True) claim a directory this way.

        Raises:
            ValueError: If the recorded settings differ from this run's.
        """
        meta_path = self.geometry_dir / FILTER_META_FILENAME
        # A disabled filter's threshold is None: it never touched the cached
        # waters, so it must not make two identical caches look incompatible.
        current = {
            "filter_by_distance": self.filter_by_distance,
            "filter_by_edia": self.filter_by_edia,
            "filter_by_bfactor": self.filter_by_bfactor,
            "max_protein_dist": self.max_protein_dist
            if self.filter_by_distance
            else None,
            "min_edia": self.min_edia if self.filter_by_edia else None,
            "max_bfactor_zscore": self.max_bfactor_zscore
            if self.filter_by_bfactor
            else None,
            "min_water_residue_ratio": self.min_water_residue_ratio,
            "max_com_dist": self.max_com_dist,
            "max_clash_fraction": self.max_clash_fraction,
            "clash_dist": self.clash_dist,
            "interface_dist_threshold": self.interface_dist_threshold,
            "cutoff": self.cutoff,
            "max_neighbors": self.max_neighbors,
        }

        if meta_path.is_file():
            with open(meta_path) as f:
                recorded = json.load(f)
            differing = [
                f"{name}: cache={recorded.get(name)!r} run={value!r}"
                for name, value in current.items()
                if recorded.get(name) != value
            ]
            if differing:
                raise ValueError(
                    f"Settings disagree with {meta_path}: {', '.join(differing)}. "
                    "The cache was filtered at write time, so one directory cannot "
                    "hold both. Match the recorded values or point "
                    "geometry_cache_name at a different directory."
                )
            return

        # Warn before writing too: stamping pre-existing entries labels them with
        # settings they were never checked against, and later runs trust the label.
        if any(self.geometry_dir.glob("*.pt")):
            logger.warning(
                f"{self.geometry_dir} has no {FILTER_META_FILENAME}; the settings "
                "its entries were built with cannot be verified."
            )

        if write:
            self.geometry_dir.mkdir(parents=True, exist_ok=True)
            # Written through a temp file: cache builds fan out over processes,
            # and a reader must never catch a half-written file.
            tmp_path = meta_path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp_path, "w") as f:
                json.dump(current, f, indent=2)
            tmp_path.replace(meta_path)

    def _preprocess_all(self):
        """
        Preprocess all PDB files that don't have cached geometry results.

        Iterates through entries, runs PyMOL crystal contact detection,
        applies quality filters, and caches results. Entries that fail
        preprocessing are logged and removed from the dataset.
        """
        self.geometry_dir.mkdir(parents=True, exist_ok=True)

        to_process = [
            e
            for e in self.entries
            if not (self.geometry_dir / f"{e['cache_key']}.pt").exists()
        ]

        if not to_process:
            logger.info("All entries already preprocessed.")
            return

        logger.info(f"Preprocessing {len(to_process)} entries...")
        failures = []
        for entry in tqdm(to_process, desc="Preprocessing"):
            cache_path = self.geometry_dir / f"{entry['cache_key']}.pt"
            try:
                self._preprocess_one(entry, cache_path)
            except Exception as e:
                logger.warning(f"\nFailed to preprocess {entry['cache_key']}: {e}")
                failures.append((entry["cache_key"], str(e)))

        # write failures to log file
        if failures:
            failure_log_path = self.geometry_dir / "preprocessing_failures.log"
            with open(failure_log_path, "a") as f:
                for pdb_id, reason in failures:
                    f.write(f"{pdb_id}\t{reason}\n")
            logger.info(f"Logged {len(failures)} failures to {failure_log_path}")

        valid_entries = [
            e
            for e in self.entries
            if (self.geometry_dir / f"{e['cache_key']}.pt").exists()
        ]
        n_removed = len(self.entries) - len(valid_entries)
        if n_removed > 0:
            logger.info(f"Filtered out {n_removed} entries without valid cache files.")
        self.entries = valid_entries
        logger.info(f"Dataset contains {len(self.entries)} valid entries.")

    def _preprocess_one(self, entry: dict, cache_path: Path):
        """
        Preprocess a single PDB file.

        Runs PyMOL crystal contact detection and caches:
        - Protein positions, features, residue indices
        - Water positions and features (if any)
        - Symmetry mate positions and features (if any)

        Raises ValueError if structure fails quality filters.
        """
        struc_path = str(entry["struc_path"])

        protein_atoms, water_atoms, ligand_atoms = parse_asu_with_biotite(struc_path)

        # check inter-chain interactions for multi-chain proteins
        chain_valid, chain_reason, _ = check_chain_interactions(
            protein_atoms,
            interface_dist_threshold=self.interface_dist_threshold,
        )
        if not chain_valid:
            raise ValueError(f"Quality filter failed: {chain_reason}")

        # PyMOL is only needed for symmetry expansion, so a no-mates cache skips
        # it (and with it the water cross-check below) entirely.
        if self.include_mates:
            crystal_data = get_crystal_contacts_pymol(
                struc_path, self.cutoff, include_ligands=self.include_ligands
            )

            # Keep only the waters PyMOL also saw. PyMOL's ASU is a superset of
            # biotite's (it keeps every altloc conformer), so a water missing from
            # it means the two parses disagree rather than that the water is
            # unwanted.
            asu_water_indices = match_atoms_to_coords(
                water_atoms, crystal_data["asu_coords"]
            )
            if asu_water_indices:
                asu_water_mask = np.zeros(len(water_atoms), dtype=bool)
                asu_water_mask[asu_water_indices] = True
                water_atoms = water_atoms[asu_water_mask]
            else:
                if len(water_atoms) > 0:
                    logger.warning(
                        f"{entry['pdb_id']}: no waters survived the biotite/PyMOL "
                        f"cross-check (had {len(water_atoms)})"
                    )
                water_atoms = water_atoms[:0]

        # Per-water filtering is optional; structure-level quality checks below always run.
        use_distance_filter = self.filter_by_distance
        use_edia_filter = self.filter_by_edia
        use_bfactor_filter = self.filter_by_bfactor
        any_filter_enabled = (
            use_distance_filter or use_edia_filter or use_bfactor_filter
        )

        if any_filter_enabled and water_atoms:
            # load EDIA data only when the EDIA filter is active
            edia_lookup = None
            if use_edia_filter:
                edia_json_path = Path(struc_path).with_suffix(".json")
                edia_lookup = load_edia_for_pdb(edia_json_path)
                if edia_lookup is None:
                    raise ValueError(
                        f"EDIA filtering enabled but JSON file missing for {entry['pdb_id']}. "
                        f"Expected file: {edia_json_path.name} in the same directory as the PDB."
                    )

            # compute normalized B-factors only when the B-factor filter is active
            # water_atoms already has b_factor from parse_asu_with_biotite — no second read needed
            bfactor_lookup = None
            if use_bfactor_filter:
                bfactor_lookup, _ = _compute_normalized_bfactors_from_atoms(water_atoms)

            # build water keys for filtering
            water_keys = list(
                zip(
                    water_atoms.chain_id.astype(str),
                    water_atoms.res_id.astype(int),
                    np.array(
                        [normalize_ins_code(x) for x in water_atoms.ins_code],
                        dtype=object,
                    ),
                )
            )

            # Apply quality filters. Mate protein atoms join the distance
            # reference so a genuine crystal-contact water is not dropped as solvent-far.
            if self.include_mates and crystal_data["mate_coords"].shape[0] > 0:
                filter_protein_coords = np.concatenate(
                    [protein_atoms.coord, crystal_data["mate_coords"]], axis=0
                )
            else:
                filter_protein_coords = protein_atoms.coord

            keep_mask = filter_waters_by_quality(
                water_atoms.coord,
                water_keys,
                filter_protein_coords if use_distance_filter else None,
                edia_lookup,
                bfactor_lookup,
                max_protein_dist=self.max_protein_dist,
                min_edia=self.min_edia,
                max_bfactor_zscore=self.max_bfactor_zscore,
                cache_key=entry["cache_key"],
            )
            water_atoms = water_atoms[keep_mask]

        protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
        water_pos_raw = (
            torch.tensor(water_atoms.coord, dtype=torch.float32)
            if water_atoms
            else torch.zeros((0, 3), dtype=torch.float32)
        )

        # Structure-level quality checks remain active even if all per-water filters are disabled.
        # check center-of-mass distance of protein atoms and water atoms (before centering)
        com_valid, com_reason = check_com_distance(
            protein_pos,
            water_pos_raw,
            max_com_dist=self.max_com_dist,
        )
        if not com_valid:
            raise ValueError(f"Quality filter failed: {com_reason}")

        # check water clashes with protein atoms
        clash_valid, clash_reason = check_water_clashes(
            protein_pos,
            water_pos_raw,
            clash_dist=self.clash_dist,
            max_clash_fraction=self.max_clash_fraction,
        )
        if not clash_valid:
            raise ValueError(f"Quality filter failed: {clash_reason}")

        # center protein positions
        center = protein_pos.mean(dim=0, keepdim=True)
        protein_pos = protein_pos - center

        protein_elements = [str(e).upper() for e in protein_atoms.element]
        protein_x = element_onehot(protein_elements)

        # protein_res_idx indexes cached ESM embedding rows, so it uses biotite's
        # residue segmentation, not res_id (not 0-based, not contiguous, repeats
        # across chains). Sanitize names and normalize ins_codes first so residues
        # split exactly where the ESM script splits them.
        sanitized_for_idx = sanitize_res_names_for_esm(protein_atoms)
        for i in range(len(sanitized_for_idx)):
            sanitized_for_idx.ins_code[i] = normalize_ins_code(
                sanitized_for_idx.ins_code[i]
            )
        num_residues = bts.get_residue_count(sanitized_for_idx)
        protein_res_idx = torch.from_numpy(
            bts.spread_residue_wise(sanitized_for_idx, np.arange(num_residues))
        ).long()

        # (chain, res_id, ins_code) -> residue index, so a symmetry mate can
        # inherit the embedding row of the ASU residue it is an image of. Keyed
        # off the same sanitized parse that defines protein_res_idx, so the index
        # lines up with the stored ESM rows.
        asu_reskey_to_residx: dict[tuple[str, int, str], int] = {}
        for res_i, start in enumerate(bts.get_residue_starts(sanitized_for_idx)):
            # ins_code already normalized in place above
            key = (
                str(sanitized_for_idx.chain_id[start]).strip(),
                int(sanitized_for_idx.res_id[start]),
                str(sanitized_for_idx.ins_code[start]),
            )
            asu_reskey_to_residx.setdefault(key, res_i)

        num_waters = len(water_atoms)
        ratio_valid, ratio_reason = check_water_residue_ratio(
            num_waters,
            num_residues,
            min_ratio=self.min_water_residue_ratio,
        )
        if not ratio_valid:
            raise ValueError(f"Quality filter failed: {ratio_reason}")

        # process water atoms
        if water_atoms:
            water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32) - center
            water_elements = [str(e).upper() for e in water_atoms.element]
            water_x = element_onehot(water_elements)
        else:
            water_pos = torch.zeros((0, 3), dtype=torch.float32)
            water_x = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)

        # Mate blocks stay empty unless include_mates (and, for ligands,
        # include_ligands) filled them in below.
        mate_pos = torch.zeros((0, 3), dtype=torch.float32)
        mate_x = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)
        mate_res_idx = torch.empty(0, dtype=torch.long)
        mate_emb_res_idx = torch.empty(0, dtype=torch.long)
        mate_lig_coords = np.zeros((0, 3), dtype=float)
        mate_lig_atoms: list = []

        if self.include_mates:
            # Drop mate atoms coincident with an ASU atom, a target water, or an
            # already-kept mate atom: special positions and redundant symmetry
            # images. Uncentered coords; mates are centered below.
            ref_parts = [protein_atoms.coord]
            if len(water_atoms):
                ref_parts.append(water_atoms.coord)
            # ASU ligands join the reference so a mate ligand that is only their
            # symmetry image goes too; neighbor-cell ligands stay.
            if self.include_ligands and len(ligand_atoms) > 0:
                ref_parts.append(ligand_atoms.coord)
            reference = np.concatenate(ref_parts, axis=0)
            mate_coords, mate_atoms = dedup_mate_atoms(
                crystal_data["mate_coords"], crystal_data["mate_atoms"], reference
            )
            # Ligand mates dedup at entity granularity, so a ligand is never
            # fragmented: whole symmetry images go, genuine neighbors stay.
            if self.include_ligands:
                mate_lig_coords, mate_lig_atoms = dedup_mate_ligands_by_residue(
                    crystal_data["mate_ligand_coords"],
                    crystal_data["mate_ligand_atoms"],
                    reference,
                )

            if mate_coords.shape[0] > 0:
                mate_pos = torch.tensor(mate_coords, dtype=torch.float32) - center
                mate_x = element_onehot([a.symbol.upper() for a in mate_atoms])

                # Group mate atoms by residue. The key omits the symmetry-object id
                # (atom.model), so two images of one residue share a group. Harmless
                # today; add atom.model before enabling GVPEncoder's pool_residue,
                # which would otherwise merge the images into one residue.
                mate_residue_keys = [(a.chain, a.resi) for a in mate_atoms]
                unique_mate_res = list(dict.fromkeys(mate_residue_keys))  # keeps order
                mate_res_map = {k: i for i, k in enumerate(unique_mate_res)}
                mate_res_idx = torch.tensor(
                    [mate_res_map[k] for k in mate_residue_keys], dtype=torch.long
                )

                # A mate inherits its ASU residue's ESM row via (chain, resi);
                # embeddings are coordinate-free, so image and source share it. -1
                # (no match) reads as a zero embedding: the atom keeps geometry and
                # element, losing only its sequence signal, so a miss warns rather
                # than raises. Misses come from PyMOL's polymer.protein admitting a
                # residue biotite dropped, or an unparseable resi.
                mate_emb_idx = []
                for atom in mate_atoms:
                    parsed = _parse_pdb_resi(atom.resi)
                    mate_emb_idx.append(
                        asu_reskey_to_residx.get((str(atom.chain).strip(), *parsed), -1)
                        if parsed is not None
                        else -1
                    )
                mate_emb_res_idx = torch.tensor(mate_emb_idx, dtype=torch.long)
                unmatched = int((mate_emb_res_idx < 0).sum())
                if unmatched:
                    logger.warning(
                        f"{entry['cache_key']}: {unmatched}/{len(mate_emb_idx)} mate "
                        "atoms unmatched to an ASU residue (zero embedding for those)"
                    )

        # Compute final protein data based on include_mates flag
        num_asu_protein = protein_pos.size(0)
        if self.include_mates and mate_pos.size(0) > 0:
            final_protein_pos = torch.cat([protein_pos, mate_pos], dim=0)
            final_protein_x = torch.cat([protein_x, mate_x], dim=0)
            # Offset mate residue indices by max protein residue index
            max_res_idx = (
                protein_res_idx.max().item() if protein_res_idx.numel() > 0 else -1
            )
            offset_mate_res_idx = mate_res_idx + max_res_idx + 1
            final_protein_res_idx = torch.cat(
                [protein_res_idx, offset_mate_res_idx], dim=0
            )
        else:
            final_protein_pos = protein_pos
            final_protein_x = protein_x
            final_protein_res_idx = protein_res_idx

        # Append ligand atoms last, giving the node order ASU protein -> mate
        # protein -> ASU ligand -> mate ligand: num_asu_protein and the mate count
        # stay meaningful, which is what keeps ESM/SLAE aligned. Ligands get
        # residue_index = emb_res_idx = -1 (no residue embedding); residue pooling
        # masks out those negatives before any scatter (GVPEncoder._pool_by_residue).
        ligand_blocks = []
        if self.include_ligands and len(ligand_atoms) > 0:
            ligand_blocks.append(
                (
                    ligand_atoms.coord,
                    [str(e).upper() for e in ligand_atoms.element],
                    False,
                )
            )
        if len(mate_lig_atoms) > 0:
            ligand_blocks.append(
                (mate_lig_coords, [a.symbol.upper() for a in mate_lig_atoms], True)
            )

        # Mate proteins inherit their source ASU residue's embedding row; ligands
        # get -1 whichever cell they came from.
        n_protein = final_protein_pos.size(0)
        emb_res_idx = torch.cat([protein_res_idx, mate_emb_res_idx], dim=0)
        is_mate = torch.zeros(n_protein, dtype=torch.bool)
        is_mate[num_asu_protein:] = True

        for coords, elements, from_mate in ligand_blocks:
            n_lig = len(elements)
            pos = torch.tensor(coords, dtype=torch.float32) - center
            final_protein_pos = torch.cat([final_protein_pos, pos], dim=0)
            final_protein_x = torch.cat(
                [final_protein_x, element_onehot(elements)], dim=0
            )
            sentinel = torch.full((n_lig,), -1, dtype=torch.long)
            final_protein_res_idx = torch.cat([final_protein_res_idx, sentinel], dim=0)
            emb_res_idx = torch.cat([emb_res_idx, sentinel], dim=0)
            is_mate = torch.cat(
                [is_mate, torch.full((n_lig,), from_mate, dtype=torch.bool)], dim=0
            )

        is_ligand = torch.zeros(final_protein_pos.size(0), dtype=torch.bool)
        is_ligand[n_protein:] = True

        # Compute PP edges and features
        if final_protein_pos.size(0) > 0:
            pp_edge_index = radius_graph(
                final_protein_pos,
                r=self.cutoff,
                loop=False,
                max_num_neighbors=self.max_neighbors,
            )
            pp_edge_index = _make_undirected(pp_edge_index)
            pp_edge_unit_vectors, pp_edge_rbf = compute_edge_features(
                final_protein_pos,
                pp_edge_index,
                num_gaussians=NUM_RBF,
                cutoff=self.cutoff,
            )
        else:
            pp_edge_index = torch.empty((2, 0), dtype=torch.long)
            pp_edge_unit_vectors, pp_edge_rbf = compute_edge_features(
                final_protein_pos,
                pp_edge_index,
                num_gaussians=NUM_RBF,
                cutoff=self.cutoff,
            )

        # Cache all data including PP edges and features
        torch.save(
            {
                "protein_pos": final_protein_pos,
                "protein_x": final_protein_x,
                "protein_res_idx": final_protein_res_idx,
                "is_ligand": is_ligand,
                "is_mate": is_mate,
                "emb_res_idx": emb_res_idx,
                "water_pos": water_pos,
                "water_x": water_x,
                # PP topology and features (precomputed)
                "pp_edge_index": pp_edge_index,
                "pp_edge_unit_vectors": pp_edge_unit_vectors,
                "pp_edge_rbf": pp_edge_rbf,
                # Metadata
                "num_asu_protein": num_asu_protein,
                "num_protein_residues": num_residues,
                "max_neighbors": self.max_neighbors,
            },
            cache_path,
        )

    def __len__(self) -> int:
        return self._effective_length

    def _annotate_data_with_embeddings(
        self,
        data: HeteroData,
        cache_key: str,
        num_asu_protein: int,
        num_protein_residues: int,
        emb_res_idx: torch.Tensor,
    ) -> None:
        """
        Load encoder-specific embeddings and attach to data object.

        Only loads embeddings for the encoder type specified at dataset init.
        GVP encoder doesn't require pre-computed embeddings. Embeddings are
        stored using generic attribute names (embedding, embedding_type) for
        consistent access regardless of encoder type.

        Args:
            data: HeteroData object to attach embeddings to (modified in-place)
            cache_key: Identifier for cached embedding files
            num_asu_protein: Number of ASU protein atoms
            num_protein_residues: Number of unique protein residues
            emb_res_idx: (N_total,) embedding row per atom -- mates inherit their
                source ASU residue's row; ligands and unmatched atoms are -1 and
                get a zero row.
        """
        if self.encoder_type == "slae":
            data["protein"].embedding = load_slae_embedding(
                embedding_dir=self.embedding_dir,
                cache_key=cache_key,
                num_asu_protein=num_asu_protein,
                total_num_atoms=data["protein"].num_nodes,
                cache_load_mmap=self.cache_load_mmap,
            )
            data["protein"].embedding_type = "slae"
        elif self.encoder_type == "esm":
            # Load residue embeddings and broadcast to atom level
            residue_embeddings = load_esm_embedding(
                embedding_dir=self.embedding_dir,
                cache_key=cache_key,
                num_protein_residues=num_protein_residues,
                cache_load_mmap=self.cache_load_mmap,
            )
            # Per-atom inheritance: a mate atom takes the row of the ASU residue
            # it images; ligands and unmatched atoms (-1) stay zero.
            atom_emb = residue_embeddings.new_zeros(
                data["protein"].num_nodes, residue_embeddings.size(1)
            )
            valid = emb_res_idx >= 0
            if valid.any():
                atom_emb[valid] = residue_embeddings[emb_res_idx[valid]]
            data["protein"].embedding = atom_emb
            data["protein"].embedding_type = "esm"

    def __getitem__(self, idx: int) -> HeteroData:
        """
        Load cached data and build graph.

        Returns HeteroData with:
        - 'protein' node type with pos, x, residue_index
        - 'water' node type with pos, x
        - ('protein', 'pp', 'protein') edges with:
            - edge_index: (2, E) topology
            - edge_unit_vectors: (E, 3) unit vectors
            - edge_rbf: (E, 16) RBF features
        - NO water edges (built dynamically in flow model)
        """
        # map idx to actual entry index (handles duplication)
        if len(self.entries) == 0:
            raise IndexError("ProteinWaterDataset is empty; no entries available.")

        actual_idx = idx % len(self.entries)
        entry = self.entries[actual_idx]
        sample_cache_key = (actual_idx, entry["cache_key"])
        if self.sample_cache_size > 0:
            cached_sample = self._sample_cache.get(sample_cache_key)
            if cached_sample is not None:
                self._sample_cache.move_to_end(sample_cache_key)
                return cached_sample.clone()

        cache_path = self.geometry_dir / f"{entry['cache_key']}.pt"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Geometry cache file not found: {cache_path}. "
                f"Run with preprocess=True to generate it."
            )

        cached = _load_torch_cache(cache_path, cache_load_mmap=self.cache_load_mmap)

        # load all data directly from cache (already includes mates if applicable)
        protein_pos = cached["protein_pos"]
        protein_x = cached["protein_x"]
        protein_res_idx = cached["protein_res_idx"]
        is_ligand = cached["is_ligand"]
        is_mate = cached["is_mate"]
        emb_res_idx = cached["emb_res_idx"]
        pp_edge_index = cached["pp_edge_index"]
        pp_edge_unit_vectors = cached["pp_edge_unit_vectors"]
        pp_edge_rbf = cached["pp_edge_rbf"]
        num_asu_protein = cached["num_asu_protein"]
        num_protein_residues = cached["num_protein_residues"]
        water_pos = cached["water_pos"]
        water_x = cached["water_x"]

        data = HeteroData()

        # compute total num_residues (protein + mates)
        num_residues = (
            int(protein_res_idx.max().item() + 1) if protein_res_idx.numel() > 0 else 0
        )

        data["protein"].x = protein_x
        data["protein"].pos = protein_pos
        data["protein"].residue_index = protein_res_idx
        data["protein"].is_ligand = is_ligand
        data["protein"].is_mate = is_mate
        data["protein"].num_nodes = protein_pos.size(0)
        data["protein"].num_residues = num_residues
        data["protein"].num_protein_residues = num_protein_residues

        self._annotate_data_with_embeddings(
            data=data,
            cache_key=entry["embedding_key"],  # use base key for embeddings
            num_asu_protein=num_asu_protein,
            num_protein_residues=num_protein_residues,
            emb_res_idx=emb_res_idx,
        )

        data["water"].x = water_x
        data["water"].pos = water_pos
        data["water"].num_nodes = water_pos.size(0)

        # load PP edges and features from cache
        data[EDGE_PP].edge_index = pp_edge_index
        data[EDGE_PP].edge_unit_vectors = pp_edge_unit_vectors
        data[EDGE_PP].edge_rbf = pp_edge_rbf

        # store metadata (use embedding_key for consistency with existing code)
        data.pdb_id = entry["embedding_key"]
        data.num_asu_protein_atoms = num_asu_protein

        if self.sample_cache_size > 0:
            self._sample_cache[sample_cache_key] = data
            self._sample_cache.move_to_end(sample_cache_key)
            while len(self._sample_cache) > self.sample_cache_size:
                self._sample_cache.popitem(last=False)
            return data.clone()

        return data


def get_dataloader(
    pdb_list_file: str,
    processed_dir: str,
    base_pdb_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    sampler: Sampler | None = None,
    distributed: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for crystal contact dataset.

    Args:
        pdb_list_file: Path to text file with PDB entries (one per line)
        processed_dir: Cache root directory. Uses:
                      - {processed_dir}/geometry for geometry caches
                      - {processed_dir}/{encoder_name} for embedding caches
        base_pdb_dir: Base directory containing PDB subdirectories
        encoder_type: Encoder used downstream ('gvp', 'slae', or 'esm').
                      Embeddings are loaded only for this type.
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of DataLoader workers (default 8)
        pin_memory: Pin memory for faster CPU-GPU transfer (default True)
        prefetch_factor: Number of batches to prefetch per worker (default 4)
        persistent_workers: Keep workers alive between epochs (default True)
        sampler: Optional sampler (e.g. DistributedSampler for DDP). When
                 provided, `shuffle` is ignored (the sampler owns ordering).
        distributed: If True (and no explicit sampler is given), build a
                 DistributedSampler from the env-configured process group so each
                 DDP rank sees a disjoint shard. Call
                 `loader.sampler.set_epoch(epoch)` each epoch to reshuffle.
        **dataset_kwargs: Additional arguments passed to ProteinWaterDataset
                         (e.g., cutoff, include_mates, duplicate_single_sample)

    Returns:
        DataLoader that yields batched HeteroData objects

    Note:
        For single-protein overfitting, use duplicate_single_sample parameter:
        - duplicate_single_sample=100 creates 100 copies of the sample in the dataset
        - Then batch_size works normally
    """
    dataset = ProteinWaterDataset(
        pdb_list_file=pdb_list_file,
        processed_dir=processed_dir,
        base_pdb_dir=base_pdb_dir,
        **dataset_kwargs,
    )

    if sampler is None and distributed:
        # drop_last=False keeps every sample; the sampler pads the last shard by
        # repeating a few, which double-counts them in epoch metrics. Accepted:
        # dropping instead gives ranks uneven shards and stalls the all-reduce.
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # A sampler is mutually exclusive with shuffle; let the sampler own ordering.
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=lambda batch: Batch.from_data_list(batch),
    )

    return loader
