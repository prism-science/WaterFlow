# inference_graph.py

"""Building the flow-model input graph from a raw PDB/CIF at inference time.

The training ProteinWaterDataset couples graph construction to ground-truth
water quality filtering and a training cache. Inference needs only the geometry
half -- protein + symmetry mates + hets, no waters. This module imports the
independent building blocks from src.dataset/src.utils and replicates the parts
of pre-processing that aren't already a standalone function.

The produced HeteroData matches what ProteinWaterDataset.__getitem__ yields
(same node order, residue indexing, PP edges and embedding attachment) but with
no water nodes. The flow model samples candidate waters itself.
"""

from __future__ import annotations

from pathlib import Path

import biotite.structure as bts
import numpy as np
import torch
from loguru import logger
from torch_cluster import radius_graph
from torch_geometric.data import HeteroData

from src.constants import DEFAULT_EDGE_CUTOFF, EDGE_PP, NODE_FEATURE_DIM, NUM_RBF
from src.dataset import (
    _make_undirected,
    _parse_pdb_resi,
    dedup_mate_atoms,
    dedup_mate_ligands_by_residue,
    element_onehot,
    get_crystal_contacts_pymol,
    load_esm_embedding,
    load_slae_embedding,
    parse_asu_with_biotite,
)
from src.utils import (
    compute_edge_features,
    normalize_ins_code,
    sanitize_res_names_for_esm,
)


def build_inference_graph(
    struc_path: str | Path,
    *,
    encoder_type: str = "gvp",
    processed_dir: str | Path | None = None,
    include_mates: bool = False,
    include_ligands: bool = True,
    cutoff: float = DEFAULT_EDGE_CUTOFF,
    max_neighbors: int = 256,
    cache_key: str | None = None,
    cache_load_mmap: bool = True,
    cache_dir: str | Path | None = None,
) -> HeteroData:
    """Parse a structure into the flow model's HeteroData, ready for sampling.

    Waters in the file are dropped; the kept atoms are protein + hets (and their
    symmetry mates when include_mates). Coordinates are centred on the ASU
    protein centroid, exactly as training preprocessing does.

    Args:
        struc_path: Path to a PDB or CIF file.
        encoder_type: "gvp" (no embeddings), "slae" or "esm" (loads a
            precomputed embedding from processed_dir/<encoder_type>).
        processed_dir: Cache root holding the embedding directory. Required for
            slae/esm; unused for gvp.
        include_mates: Add crystallographic symmetry mates (PyMOL symexp).
        include_ligands: Keep hetero atoms (ligands, ions, cofactors, nucleic
            acids) and their mates.
        cutoff, max_neighbors: PP radius-graph parameters (match the flow run).
        cache_key: Key for the embedding lookup. Defaults to the file stem
            (e.g. 6eey_final).
        cache_dir: If given, the graph is loaded from cache_dir/<cache_key>.pt
            when present, else built and saved there. Stored graphs hold no
            embeddings; those are attached from processed_dir on every call.
            The file records nothing about the build settings, so a cache_dir
            must only be shared by calls with identical settings.

    Returns:
        HeteroData with centred protein nodes (+ optional embeddings), empty
        water nodes, cached PP edges, and .center, the (3,) ASU protein
        centroid that coordinates were shifted by.
    """
    struc_path = str(struc_path)
    if cache_key is None:
        cache_key = Path(struc_path).stem
    if encoder_type not in ("gvp", "slae", "esm"):
        raise ValueError(f"encoder_type={encoder_type!r} must be one of gvp, slae, esm")
    if encoder_type in ("slae", "esm") and processed_dir is None:
        raise ValueError(
            f"encoder_type={encoder_type!r} needs precomputed embeddings; pass "
            "processed_dir (its <encoder_type> subdir). Run generate_"
            f"{encoder_type}_embeddings.py first."
        )

    graph_path = Path(cache_dir) / f"{cache_key}.pt" if cache_dir is not None else None
    if graph_path is not None and graph_path.exists():
        data = torch.load(graph_path, weights_only=False)
        _attach_embeddings(
            data,
            encoder_type=encoder_type,
            processed_dir=processed_dir,
            cache_load_mmap=cache_load_mmap,
        )
        return data

    # Waters are prediction targets, not inputs -- drop them and keep protein + hets.
    protein_atoms, _waters, ligand_atoms = parse_asu_with_biotite(struc_path)
    if protein_atoms.array_length() == 0:
        raise ValueError(f"No protein atoms parsed from {struc_path}")

    crystal_data = None
    if include_mates:
        crystal_data = get_crystal_contacts_pymol(
            struc_path, cutoff, include_ligands=include_ligands
        )

    protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
    center = protein_pos.mean(dim=0, keepdim=True)
    protein_pos = protein_pos - center

    protein_x = element_onehot([str(e).upper() for e in protein_atoms.element])

    protein_res_idx, num_residues, asu_reskey_to_residx = _residue_indexing(
        protein_atoms
    )

    (
        mate_pos,
        mate_x,
        mate_res_idx,
        mate_emb_res_idx,
        mate_lig_coords,
        mate_lig_atoms,
    ) = _build_mates(
        crystal_data,
        center=center,
        reference_atoms=protein_atoms,
        ligand_atoms=ligand_atoms if include_ligands else None,
        asu_reskey_to_residx=asu_reskey_to_residx,
        cache_key=cache_key,
    )

    num_asu_protein = protein_pos.size(0)
    if mate_pos.size(0) > 0:
        final_pos = torch.cat([protein_pos, mate_pos], dim=0)
        final_x = torch.cat([protein_x, mate_x], dim=0)
        max_res_idx = protein_res_idx.max().item() if protein_res_idx.numel() else -1
        final_res_idx = torch.cat(
            [protein_res_idx, mate_res_idx + max_res_idx + 1], dim=0
        )
    else:
        final_pos, final_x, final_res_idx = protein_pos, protein_x, protein_res_idx

    n_protein = final_pos.size(0)
    emb_res_idx = torch.cat([protein_res_idx, mate_emb_res_idx], dim=0)
    is_mate = torch.zeros(n_protein, dtype=torch.bool)
    is_mate[num_asu_protein:] = True

    # Ligand blocks last -> node order [ASU protein | mate protein | ASU lig | mate lig].
    ligand_blocks = []
    if include_ligands and len(ligand_atoms) > 0:
        ligand_blocks.append(
            (ligand_atoms.coord, [str(e).upper() for e in ligand_atoms.element], False)
        )
    if len(mate_lig_atoms) > 0:
        ligand_blocks.append(
            (mate_lig_coords, [a.symbol.upper() for a in mate_lig_atoms], True)
        )

    for coords, elements, from_mate in ligand_blocks:
        n_lig = len(elements)
        pos = torch.tensor(coords, dtype=torch.float32) - center
        final_pos = torch.cat([final_pos, pos], dim=0)
        final_x = torch.cat([final_x, element_onehot(elements)], dim=0)
        sentinel = torch.full((n_lig,), -1, dtype=torch.long)
        final_res_idx = torch.cat([final_res_idx, sentinel], dim=0)
        emb_res_idx = torch.cat([emb_res_idx, sentinel], dim=0)
        is_mate = torch.cat(
            [is_mate, torch.full((n_lig,), from_mate, dtype=torch.bool)], dim=0
        )

    is_ligand = torch.zeros(final_pos.size(0), dtype=torch.bool)
    is_ligand[n_protein:] = True

    pp_edge_index, pp_unit_vectors, pp_rbf = _pp_edges(
        final_pos, cutoff=cutoff, max_neighbors=max_neighbors
    )

    data = _assemble_hetero(
        final_pos=final_pos,
        final_x=final_x,
        final_res_idx=final_res_idx,
        is_ligand=is_ligand,
        is_mate=is_mate,
        emb_res_idx=emb_res_idx,
        num_asu_protein=num_asu_protein,
        num_protein_residues=num_residues,
        pp_edge_index=pp_edge_index,
        pp_unit_vectors=pp_unit_vectors,
        pp_rbf=pp_rbf,
        pdb_id=cache_key,
    )

    # ASU protein centroid; adding it back returns predictions to the input frame.
    data.center = center.squeeze(0)  # (3,)

    # Saved before embeddings are attached, so the file stays small.
    if graph_path is not None:
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, graph_path)

    _attach_embeddings(
        data,
        encoder_type=encoder_type,
        processed_dir=processed_dir,
        cache_load_mmap=cache_load_mmap,
    )
    return data


def _residue_indexing(
    protein_atoms: bts.AtomArray,
) -> tuple[torch.Tensor, int, dict[tuple[str, int, str], int]]:
    """ESM-aligned per-atom residue index, residue count, and a reskey->index map.

    Residues are split off the same sanitized parse the ESM script uses, so the
    indices line up with the stored embedding rows.
    """
    sanitized = sanitize_res_names_for_esm(protein_atoms)
    for i in range(len(sanitized)):
        sanitized.ins_code[i] = normalize_ins_code(sanitized.ins_code[i])

    num_residues = bts.get_residue_count(sanitized)
    res_idx = torch.from_numpy(
        bts.spread_residue_wise(sanitized, np.arange(num_residues))
    ).long()

    reskey_to_residx: dict[tuple[str, int, str], int] = {}
    for res_i, start in enumerate(bts.get_residue_starts(sanitized)):
        key = (
            str(sanitized.chain_id[start]).strip(),
            int(sanitized.res_id[start]),
            str(sanitized.ins_code[start]),
        )
        reskey_to_residx.setdefault(key, res_i)
    return res_idx, num_residues, reskey_to_residx


def _build_mates(
    crystal_data,
    *,
    center: torch.Tensor,
    reference_atoms: bts.AtomArray,
    ligand_atoms: bts.AtomArray | None,
    asu_reskey_to_residx: dict[tuple[str, int, str], int],
    cache_key: str,
):
    """Deduped, centred symmetry-mate tensors, mirroring _preprocess_one.

    Returns (mate_pos, mate_x, mate_res_idx, mate_emb_res_idx, mate_lig_coords,
    mate_lig_atoms). All empty when there are no mates.
    """
    mate_pos = torch.zeros((0, 3), dtype=torch.float32)
    mate_x = torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float32)
    mate_res_idx = torch.empty(0, dtype=torch.long)
    mate_emb_res_idx = torch.empty(0, dtype=torch.long)
    mate_lig_coords = np.zeros((0, 3), dtype=float)
    mate_lig_atoms: list = []

    if crystal_data is None:
        return (
            mate_pos,
            mate_x,
            mate_res_idx,
            mate_emb_res_idx,
            mate_lig_coords,
            mate_lig_atoms,
        )

    # Drop mate atoms coincident with an ASU atom or an already-kept mate image.
    # Deliberate divergence from training: _preprocess_one also puts the ASU
    # (filtered) waters in this reference, so a mate on a special position atop a
    # water is dropped there. Inference has no waters, so such a mate is kept --
    # a rare special-position case with negligible effect on the graph.
    ref_parts = [reference_atoms.coord]
    if ligand_atoms is not None and len(ligand_atoms) > 0:
        ref_parts.append(ligand_atoms.coord)
    reference = np.concatenate(ref_parts, axis=0)

    mate_coords, mate_atoms = dedup_mate_atoms(
        crystal_data["mate_coords"], crystal_data["mate_atoms"], reference
    )
    if ligand_atoms is not None:
        mate_lig_coords, mate_lig_atoms = dedup_mate_ligands_by_residue(
            crystal_data["mate_ligand_coords"],
            crystal_data["mate_ligand_atoms"],
            reference,
        )

    if mate_coords.shape[0] > 0:
        mate_pos = torch.tensor(mate_coords, dtype=torch.float32) - center
        mate_x = element_onehot([a.symbol.upper() for a in mate_atoms])

        mate_residue_keys = [(a.chain, a.resi) for a in mate_atoms]
        unique_res = list(dict.fromkeys(mate_residue_keys))  # order-preserving
        res_map = {k: i for i, k in enumerate(unique_res)}
        mate_res_idx = torch.tensor(
            [res_map[k] for k in mate_residue_keys], dtype=torch.long
        )

        # A mate inherits its ASU residue's embedding row via (chain, resi); -1
        # (no match) reads as a zero row, so a miss warns rather than raises.
        emb_idx = []
        for atom in mate_atoms:
            parsed = _parse_pdb_resi(atom.resi)
            emb_idx.append(
                asu_reskey_to_residx.get((str(atom.chain).strip(), *parsed), -1)
                if parsed is not None
                else -1
            )
        mate_emb_res_idx = torch.tensor(emb_idx, dtype=torch.long)
        unmatched = int((mate_emb_res_idx < 0).sum())
        if unmatched:
            logger.warning(
                f"{cache_key}: {unmatched}/{len(emb_idx)} mate atoms unmatched to "
                "an ASU residue (zero embedding for those)"
            )

    return (
        mate_pos,
        mate_x,
        mate_res_idx,
        mate_emb_res_idx,
        mate_lig_coords,
        mate_lig_atoms,
    )


def _pp_edges(pos: torch.Tensor, *, cutoff: float, max_neighbors: int):
    """Undirected protein-protein radius graph plus its edge features."""
    if pos.size(0) > 0:
        edge_index = radius_graph(
            pos, r=cutoff, loop=False, max_num_neighbors=max_neighbors
        )
        edge_index = _make_undirected(edge_index)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    unit_vectors, rbf = compute_edge_features(
        pos, edge_index, num_gaussians=NUM_RBF, cutoff=cutoff
    )
    return edge_index, unit_vectors, rbf


def _assemble_hetero(
    *,
    final_pos,
    final_x,
    final_res_idx,
    is_ligand,
    is_mate,
    emb_res_idx,
    num_asu_protein,
    num_protein_residues,
    pp_edge_index,
    pp_unit_vectors,
    pp_rbf,
    pdb_id,
) -> HeteroData:
    """Build the HeteroData, mirroring __getitem__'s node layout."""
    data = HeteroData()

    num_residues = int(final_res_idx.max().item() + 1) if final_res_idx.numel() else 0

    data["protein"].x = final_x
    data["protein"].pos = final_pos
    data["protein"].residue_index = final_res_idx
    data["protein"].is_ligand = is_ligand
    data["protein"].is_mate = is_mate
    data["protein"].num_nodes = final_pos.size(0)
    data["protein"].num_residues = num_residues
    data["protein"].num_protein_residues = num_protein_residues
    # Per-atom row into the residue embedding table (-1 = no row: ligands and
    # unmatched mates), kept so embeddings can be attached to a loaded graph.
    data["protein"].emb_res_idx = emb_res_idx

    # Empty water nodes: the flow model samples candidates itself.
    data["water"].x = torch.zeros((0, NODE_FEATURE_DIM), dtype=torch.float32)
    data["water"].pos = torch.zeros((0, 3), dtype=torch.float32)
    data["water"].num_nodes = 0

    data[EDGE_PP].edge_index = pp_edge_index
    data[EDGE_PP].edge_unit_vectors = pp_unit_vectors
    data[EDGE_PP].edge_rbf = pp_rbf

    data.pdb_id = pdb_id
    data.num_asu_protein_atoms = num_asu_protein
    return data


def _attach_embeddings(
    data: HeteroData,
    *,
    encoder_type: str,
    processed_dir,
    cache_load_mmap: bool,
) -> None:
    """Load and attach cached embeddings, mirroring _annotate_data_with_embeddings."""
    if encoder_type == "gvp":
        return
    embedding_dir = Path(processed_dir) / encoder_type
    cache_key = data.pdb_id

    if encoder_type == "slae":
        data["protein"].embedding = load_slae_embedding(
            embedding_dir=embedding_dir,
            cache_key=cache_key,
            num_asu_protein=int(data.num_asu_protein_atoms),
            total_num_atoms=data["protein"].num_nodes,
            cache_load_mmap=cache_load_mmap,
        )
        data["protein"].embedding_type = "slae"
    elif encoder_type == "esm":
        residue_embeddings = load_esm_embedding(
            embedding_dir=embedding_dir,
            cache_key=cache_key,
            num_protein_residues=int(data["protein"].num_protein_residues),
            cache_load_mmap=cache_load_mmap,
        )
        emb_res_idx = data["protein"].emb_res_idx
        atom_emb = residue_embeddings.new_zeros(
            data["protein"].num_nodes, residue_embeddings.size(1)
        )
        valid = emb_res_idx >= 0
        if valid.any():
            atom_emb[valid] = residue_embeddings[emb_res_idx[valid]]
        data["protein"].embedding = atom_emb
        data["protein"].embedding_type = "esm"
