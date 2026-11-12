"""
Generate and cache ESM3 residue-level embeddings for protein structures.

This script:
1. Reads a split file with PDB entries (format: 'pdbid_final' or 'pdbid_final_chain')
2. Loads the PDB using Biotite to extract the ground-truth atoms and sequence.
3. Sanitizes the AtomArray in-memory (removes HETATM flags, standardizes residue
   names) so ESM3 structurally embeds non-canonical residues (mapped to 'UNK'/'X').
4. Runs ESM3 to get per-residue embeddings.
5. Saves embeddings to separate .pt files in processed_dir/esm/

Usage:
    # training layout: keyed by split entry ('6eey_final' -> esm/6eey_final.pt)
    uv run python -m scripts.generate_esm_embeddings \
        --pdb_list /path/to/split.txt \
        --processed_dir /path/to/cache \
        --base_pdb_dir /path/to/pdbs \
        [--device cuda:0] \
        [--model_name esm3-open] \
        [--batch_limit 10] \
        [--overwrite]

    # raw structures: keyed by file stem (protein.cif -> esm/protein.pt), which
    # is how predict_waters looks embeddings up
    uv run python -m scripts.generate_esm_embeddings \
        --struc protein.cif other.pdb \
        --processed_dir /path/to/cache

Output format (per cache file):
    {
        'residue_embeddings': Tensor of shape (N_res, embed_dim),
        'sequence': str,  # biotite-derived sequence (X for non-canonical)
        'pdb_id': str,
        'num_residues': int,
    }
"""

import argparse
import io
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from biotite.structure.io.pdb import PDBFile
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_complex import ProteinComplex
from loguru import logger
from tqdm import tqdm

from src.constants import THREE_TO_ONE
from src.dataset import parse_asu_with_biotite
from src.utils import (
    normalize_ins_code,
    parse_split_file,
    sanitize_res_names_for_esm,
    setup_logging_for_tqdm,
)


@contextmanager
def default_dtype(dtype: torch.dtype):
    """Temporarily change the dtype that new tensors are created with."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_esm3_model(model_name: str, device: torch.device) -> ESM3:
    """Load ESM3 without the float32 spike ``from_pretrained`` incurs.

    ``ESM3.from_pretrained`` builds the model in float32 directly on the
    target device (``with torch.device(device): ESM3(...)``, in
    ``esm/pretrained.py``), loads the bfloat16 checkpoint onto that same
    device, and only afterwards casts the model to bfloat16. The float32
    model and the checkpoint are briefly resident together, so the load
    transiently needs ~7.9 GiB of VRAM for a model that occupies ~2.6 GiB
    once settled. That exceeds what a nominally 8 GB card provides, and the
    load fails before any structure has been read.

    On CUDA, constructing in bfloat16 -- the dtype the model is cast to
    anyway -- lowers the peak to ~5.3 GiB at no cost in host memory. The
    resulting weights and embeddings are bitwise identical.

    On CPU the model stays float32, matching what ``from_pretrained`` does
    there today. Passing the device explicitly also stops a CPU run from
    allocating on the GPU merely because one is visible.
    """
    if device.type == "cpu":
        return ESM3.from_pretrained(model_name, device=device)
    with default_dtype(torch.bfloat16):
        return ESM3.from_pretrained(model_name, device=device)


def compute_esm_embeddings(
    struc_path: Path,
    model: ESM3,
) -> dict | None:
    """
    Compute ESM3 residue-level embeddings using an in-memory sanitized structure.

    This bypasses ESM's default behavior of dropping HETATM non-canonicals.
    It extracts the sequence, strips HETATM flags, renames modified residues
    (e.g., MSE -> MET) and unknowns (-> UNK), and feeds the buffer directly
    to ESM3 so that all residues receive a structural embedding.

    How ESM parses: (https://github.com/evolutionaryscale/esm/blob/main/esm/utils/structure/protein_chain.py)

    Args:
        struc_path: Path to structure file (PDB/CIF)
        model: Loaded ESM3 model

    Returns:
        Dict with 'residue_embeddings', 'sequence', 'num_residues', or None on error
    """
    try:
        # Load ground truth atoms using geometry cache parser in src/dataset.py
        protein_atoms, _, _ = parse_asu_with_biotite(str(struc_path))
        if len(protein_atoms) == 0:
            raise ValueError(f"No protein atoms found in {struc_path}")

        # Extract ground truth sequence before mutating the array
        key_to_resname = {}

        chain_id_arr = protein_atoms.chain_id
        res_id_arr = protein_atoms.res_id
        res_name_arr = protein_atoms.res_name
        ins_code_arr = np.array(
            [normalize_ins_code(x) for x in protein_atoms.ins_code], dtype=object
        )

        for i in range(len(protein_atoms)):
            key = (chain_id_arr[i], res_id_arr[i], ins_code_arr[i])
            if key not in key_to_resname:
                key_to_resname[key] = res_name_arr[i]

        unique_res_keys = list(key_to_resname.keys())

        biotite_seq = [
            THREE_TO_ONE.get(key_to_resname[key], "X") for key in unique_res_keys
        ]
        num_residues = len(biotite_seq)

        # Sanitize the AtomArray so ESM accepts all residues. Uses the shared
        # helper so residue-name canonicalization stays identical to the residue
        # indexing in src/dataset.py.
        protein_atoms = sanitize_res_names_for_esm(protein_atoms)
        protein_atoms.hetero[:] = False

        # Write sanitized array to an in-memory buffer
        sanitized_pdb = PDBFile()
        sanitized_pdb.set_structure(protein_atoms)
        buf = io.StringIO()
        sanitized_pdb.write(buf)
        buf.seek(0)

        # Run ESM3 inference on the sanitized structure
        complex_obj = ProteinComplex.from_pdb(buf)
        protein = ESMProtein.from_protein_complex(complex_obj)

        if not protein.sequence or protein.sequence.replace("|", "") == "":
            raise ValueError(f"ESM returned empty sequence for {struc_path}")

        with torch.no_grad():
            protein_tensor = model.encode(protein)
            output = model.logits(
                protein_tensor,
                LogitsConfig(return_embeddings=True),
            )

        emb = output.embeddings
        if emb.dim() == 3:
            emb = emb.squeeze(0)
        emb = emb[1:-1].detach().cpu()  # remove BOS/EOS tokens

        # Remove chain breaks and validate length
        esm_seq_with_breaks = protein.sequence
        is_chainbreak = torch.tensor(
            [aa == "|" for aa in esm_seq_with_breaks], dtype=torch.bool
        )
        esm_emb = emb[~is_chainbreak]
        esm_seq = "".join([aa for aa in esm_seq_with_breaks if aa != "|"])

        # Validate: length mismatch means embeddings won't align with residues
        if len(esm_seq) != num_residues:
            raise ValueError(
                f"Length mismatch after sanitization for {struc_path}! "
                f"Biotite: {num_residues}, ESM: {len(esm_seq)}"
            )

        return {
            "residue_embeddings": esm_emb,
            "sequence": "".join(biotite_seq),
            "num_residues": num_residues,
        }

    except Exception as e:
        logger.error(f"Error computing embeddings for {struc_path}: {e}")
        return None


def main() -> None:
    setup_logging_for_tqdm()
    parser = argparse.ArgumentParser(
        description="Generate ESM3 embeddings for protein structures"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pdb_list",
        type=Path,
        help="Text file with PDB entries (one per line, e.g., '6eey_final'); "
        "resolved under --base_pdb_dir and keyed by split entry.",
    )
    src.add_argument(
        "--struc",
        type=Path,
        nargs="+",
        help="One or more raw PDB/CIF files, keyed by file stem "
        "(protein.cif -> esm/protein.pt). Use this for prediction inputs.",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
        help="Base cache directory; embeddings saved to {processed_dir}/esm/",
    )
    parser.add_argument(
        "--base_pdb_dir",
        type=Path,
        default=None,
        help="Base directory containing PDB subdirectories; required with --pdb_list",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="esm3-open",
        help="ESM3 model name to load (default: esm3-open)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for ESM model",
    )
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embeddings",
    )

    args = parser.parse_args()
    if args.pdb_list and args.base_pdb_dir is None:
        parser.error("--pdb_list requires --base_pdb_dir")
    if args.struc:
        counts = Counter(p.stem for p in args.struc)
        duplicates = sorted(s for s, n in counts.items() if n > 1)
        if duplicates:
            parser.error(
                f"duplicate structure names {duplicates}: embeddings are keyed "
                "by file stem, so every --struc file needs a distinct name"
            )

    esm_cache_dir = args.processed_dir / "esm"
    esm_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.struc:
        # Key by file stem, matching how predict_waters looks embeddings up.
        entries = [
            {"cache_key": p.stem, "struc_path": p, "pdb_id": p.stem} for p in args.struc
        ]
        logger.info(f"Embedding {len(entries)} structure(s) passed via --struc")
    else:
        entries = parse_split_file(args.pdb_list, args.base_pdb_dir)
        logger.info(f"Found {len(entries)} entries in split file")

    if args.batch_limit:
        entries = entries[: args.batch_limit]
        logger.info(f"Processing first {args.batch_limit} entries")

    if not args.overwrite:
        to_process = []
        for entry in entries:
            cache_path = esm_cache_dir / f"{entry['cache_key']}.pt"
            if not cache_path.exists():
                to_process.append(entry)
        logger.info(f"{len(to_process)} entries need ESM embeddings")
        entries = to_process

    if not entries:
        logger.info("No entries to process. Done.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading ESM3 model: {args.model_name}...")
    model = load_esm3_model(args.model_name, device)
    model.eval()
    logger.info("Model loaded.")

    success_count = 0
    failures = []

    for entry in tqdm(entries, desc="Computing ESM embeddings"):
        cache_key = entry["cache_key"]
        cache_path = esm_cache_dir / f"{cache_key}.pt"

        result = compute_esm_embeddings(entry["struc_path"], model)

        if result is not None:
            result["pdb_id"] = entry["pdb_id"]
            torch.save(result, cache_path)
            success_count += 1
        else:
            failures.append((cache_key, "Embedding computation failed"))

    logger.info(
        f"Completed: {success_count}/{len(entries)} entries processed successfully"
    )

    if failures:
        failure_log_path = esm_cache_dir / "embedding_failures.log"
        with open(failure_log_path, "a") as f:
            for cache_key, reason in failures:
                f.write(f"{cache_key}\t{reason}\n")
        logger.error(f"Logged {len(failures)} failures to {failure_log_path}")


if __name__ == "__main__":
    main()
