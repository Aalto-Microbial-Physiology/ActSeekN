import os
import time
import logging
import argparse
import pickle
import re
import sys
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
import requests
from Bio.PDB import *
import subprocess
import concurrent.futures
from dataclasses import dataclass
import ActSeekLib


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("requests.packages.urllib3").setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
STRUCTURES_DIR = ROOT_DIR / "structures"
RESULTS_DIR = ROOT_DIR / "results"
ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb"

aa_codes = {
    "ALA": 1, "ARG": 2, "ASN": 3, "ASP": 4, "CYS": 5,
    "GLN": 6, "GLU": 7, "GLY": 8, "HIS": 9, "ILE": 10,
    "LEU": 11, "LYS": 12, "MET": 13, "PHE": 14, "PRO": 15,
    "SER": 16, "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20,
    "SEC": 21
    }

def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_h5_mapping(group):
    keys = group["keys"][()]
    values = group["values"][()]
    mapping = {}
    for key, value in zip(keys.tolist(), values.tolist()):
        decoded = _decode_scalar(value)
        mapping[int(key)] = decoded
    return mapping


def _normalize_index_used(index_used, active_len, cavity_len, cb_len):
    max_valid = min(active_len, cavity_len, cb_len)
    if max_valid <= 0:
        return None

    if all(0 <= idx < max_valid for idx in index_used):
        return index_used

    one_based = [idx - 1 for idx in index_used]
    if all(0 <= idx < max_valid for idx in one_based):
        return one_based

    return None


def _load_h5_entry(group_name, group):
    seed_coords = np.asarray(group["search_protein_coords"][()])
    cavity_coords = np.asarray(group["search_amino_acid_coords_all"][()])
    cavity_coords_cb = np.asarray(group["search_amino_acids_betac_coords_all"][()])
    active = np.asarray(group["search_amino_acid_indexes"][()], dtype=int)
    active_indices_raw = [int(x) for x in np.asarray(group["search_index_used"][()]).tolist()]
    real_index_seed = _read_h5_mapping(group["search_real_index"])

    if len(active) != len(cavity_coords) or len(active) != len(cavity_coords_cb):
        raise ValueError(
            f"length mismatch: active={len(active)}, cavity={len(cavity_coords)}, cb={len(cavity_coords_cb)}"
        )

    active_indices = _normalize_index_used(active_indices_raw, len(active), len(cavity_coords), len(cavity_coords_cb))
    if active_indices is None:
        raise ValueError(
            f"invalid search_index_used {active_indices_raw} for lengths active={len(active)}, cavity={len(cavity_coords)}, cb={len(cavity_coords_cb)}"
        )

    missing_real_index = sorted({int(x) for x in active.tolist() if int(x) not in real_index_seed})
    if missing_real_index:
        raise ValueError(
            f"active residue ids missing from search_real_index: {missing_real_index[:10]}"
        )

    return {
        "ec_number": group_name.rsplit("-", 1)[0],
        "seed_coords": seed_coords,
        "cavity_coords": cavity_coords,
        "cavity_coords_cb": cavity_coords_cb,
        "aa_cavity": _read_h5_mapping(group["amino_acid_group"]),
        "active": active,
        "real_index_seed": real_index_seed,
        "protein_id": str(_decode_scalar(group["seed_protein"][()])),
        "active_indices": np.asarray(active_indices, dtype=int),
        "score": int(float(_decode_scalar(group["score"][()]))),
        "description": str(_decode_scalar(group["description"][()])),
    }


def _read_database_entries(h5_path):
    @dataclass
    class database_entry2:
        ec_number: str
        seed_coords: np.ndarray
        cavity_coords: np.ndarray
        cavity_coords_cb: np.ndarray
        aa_cavity: dict
        active: np.ndarray
        real_index_seed: dict
        protein_id: str
        active_indices: np.ndarray
        score: int
        description: str

    entries = []
    skipped_entries = 0
    with h5py.File(h5_path, "r") as handle:
        for group_name in handle.keys():
            group = handle[group_name]
            try:
                entry_data = _load_h5_entry(group_name, group)
            except Exception as exc:
                skipped_entries += 1
                logging.warning(f"Skipping malformed HDF5 entry {group_name}: {exc}")
                continue

            entries.append(database_entry2(**entry_data))
    if skipped_entries:
        logging.warning(f"Skipped {skipped_entries} malformed entries while loading {h5_path.name}.")
    return entries


def _resolve_database_path():
    preferred = DATA_DIR / "distances_withAI_and_desc_05_03_2026.h5"
    if preferred.is_file():
        return preferred

    candidates = sorted(DATA_DIR.glob("distances_with*AI*.h5"))
    if candidates:
        return candidates[-1]

    fallback = DATA_DIR / "newDatabase_v3.pickle"
    if fallback.is_file():
        return fallback

    raise FileNotFoundError("No supported annotation database found in data/.")


def _build_metadata_lookup(distances):
    metadata_lookup = {}
    for data in distances:
        metadata_lookup[f"{data.protein_id} -- {data.ec_number}"] = {
            "Confidence": getattr(data, "score", None),
            "Description": getattr(data, "description", ""),
        }
    return metadata_lookup


def _clean_optional_text(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _sanitize_filename_part(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._-") or "row"


def _build_result_filename(case_request, resolved_entry):
    original_label = _sanitize_filename_part(case_request["requested_entry"])
    return f"actseekn-{original_label}-results.csv"


def _find_existing_result(case_request, results_dir):
    requested_entry = case_request.get("requested_entry")
    if not requested_entry:
        return None, None, None

    original_label = _sanitize_filename_part(requested_entry)

    result_file = results_dir / f"actseekn-{original_label}-results.csv"
    if result_file.is_file():
        return result_file, original_label
    return None, None


def _build_case_result(case_request, success, resolved_uniprot=None):
    return {
        "success": success,
        "row_index": case_request["row_index"],
        "resolved_uniprot": resolved_uniprot,
        "original_uniprot": case_request.get("requested_entry"),
    }


def _build_case_requests(input_file):
    table = pd.read_csv(input_file)
    if "Entry" not in table.columns:
        raise ValueError(f"Input file {input_file} must contain an 'Entry' column.")

    case_requests = []
    skipped_row_indices = []
    skipped_rows = 0

    for row_index, row in table.iterrows():
        entry = _clean_optional_text(row.get("Entry"))
        label = _sanitize_filename_part(entry or f"row{row_index + 1}")
        if not entry:
            skipped_rows += 1
            skipped_row_indices.append(row_index)
            logging.warning(f"Skipping row {row_index + 1}: empty Entry provided.")
            continue

        case_request = {
            "row_index": row_index,
            "query_label": label,
            "requested_entry": entry,
        }
        case_requests.append(case_request)

    logging.info(
        f"Prepared {len(case_requests)} query proteins from {input_file}; skipped {skipped_rows} rows."
    )
    return case_requests, table, skipped_row_indices


def _write_failed_input_rows(input_file, input_table, failed_row_indices):
    failed_rows = sorted(set(failed_row_indices))
    output_path = RESULTS_DIR / f"{input_file.stem}_failed.csv"
    input_table.iloc[failed_rows].to_csv(output_path, index=False)
    logging.info(f"Wrote {len(failed_rows)} failed input rows to {output_path}.")


def _is_nonempty_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False

    suffix = path.suffix.lower()
    parser = MMCIFParser(QUIET=True) if suffix == ".cif" else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("validation", str(path))
    except Exception:
        return False

    for atom in structure.get_atoms():
        if atom.get_name().strip() in {"CA", "CB", "N", "C", "O"}:
            return True
    return False


def _download_structure(accession):
    structure_path = STRUCTURES_DIR / f"AF-{accession}-F1-model_v6.pdb"
    if _is_nonempty_file(structure_path):
        return structure_path

    response = requests.get(
        ALPHAFOLD_PDB_URL.format(accession=accession),
        timeout=60,
        stream=True,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    with open(structure_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return structure_path


def _resolve_case_structure(case_request):
    requested_entry = case_request.get("requested_entry")
    if not requested_entry:
        raise FileNotFoundError("No UniProt entry provided.")

    af_structure_path = STRUCTURES_DIR / f"AF-{requested_entry}-F1-model_v6.pdb"
    if _is_nonempty_file(af_structure_path):
        return requested_entry, af_structure_path, None

    local_pdb_path = STRUCTURES_DIR / f"{requested_entry}.pdb"
    if _is_nonempty_file(local_pdb_path):
        return requested_entry, local_pdb_path, None

    local_cif_path = STRUCTURES_DIR / f"{requested_entry}.cif"
    if _is_nonempty_file(local_cif_path):
        return requested_entry, local_cif_path, None

    try:
        structure_path = _download_structure(requested_entry)
    except Exception as exc:
        raise FileNotFoundError(f"Structure download failed for {requested_entry}: {exc}") from exc
    if structure_path is not None:
        return requested_entry, structure_path, None

    raise FileNotFoundError(
        f"No structure found for {requested_entry} in structures/ or AlphaFold DB."
    )


def read_pdbs(case_protein):
    parser = MMCIFParser() if Path(case_protein).suffix.lower() == ".cif" else PDBParser()
    case_structure = parser.get_structure("complex2", case_protein)
    aa = {}
    protein_coords = []
    protein_coords_cb = []
    i = 0
    real_index = {}
    for model in case_structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if "CA" in atom.fullname:
                        protein_coords.append(atom.get_coord())
                    if 'CB' in atom.fullname:
                        protein_coords_cb.append(atom.get_coord())
                if residue.get_resname() == "GLY":
                    protein_coords_cb.append([-10000000, -10000000, -10000000])
                aa[int(residue.get_id()[1])] = str(residue.get_resname())
                real_index[i] = int(residue.get_id()[1])
                i = i + 1
            break
    return protein_coords, protein_coords_cb, aa, real_index


def calc(case_request, segment_name, table_name, results_dir, metadata_lookup):
    try:
        query_label = case_request["query_label"]
        requested_entry = case_request["requested_entry"]
        logging.debug(f'{query_label} started...')

        existing_result, cached_resolved_entry = _find_existing_result(case_request, results_dir)
        if existing_result is not None:
            logging.debug(f'{query_label} had existing results.')
            return _build_case_result(case_request, True, cached_resolved_entry or requested_entry)

        resolved_entry, case_protein_filepath, _ = _resolve_case_structure(case_request)
        result_file = results_dir / _build_result_filename(case_request, resolved_entry)
        if result_file.is_file():
            logging.debug(f'{query_label} had existing results.')
            return _build_case_result(case_request, True, resolved_entry)
        if requested_entry and requested_entry == resolved_entry:
            logging.debug(f"{query_label} used UniProt entry {resolved_entry}.")

        case_protein_profile = [resolved_entry] + list(read_pdbs(str(case_protein_filepath)))

        start_time = time.perf_counter()
        results = ActSeekLib.concurrentMain(case_protein_profile, segment_name, table_name)
        dataset = pd.DataFrame(filter(None, results), columns=['ID', 'Length hit', 'Length query', 'Mapping', 'Distances', 'Av dist', 'Arround dist', 'Structure score', 'RMSDmin','Percentage', 'EC'])
        metadata = dataset["ID"].map(metadata_lookup).apply(pd.Series)
        dataset["Confidence"] = metadata.get("Confidence")
        dataset["Description"] = metadata.get("Description")
        dataset = dataset.sort_values(by=['RMSDmin']).round(6)
        dataset.to_csv(result_file, index=False)
        logging.debug(f"{query_label} took {time.perf_counter() - start_time:.2f}s to find {len(results)} results.")
        return _build_case_result(case_request, True, resolved_entry)
    except Exception as e:
        logging.error(f"{case_request.get('query_label', 'query')} failed with {e}")
        return _build_case_result(case_request, False, None)


def check_gpu_availability():
    if not ActSeekLib.gpu_enabled():
        logging.debug("GPU explicitly disabled, using CPU only.")
        return False
    try:
        subprocess.check_output('nvidia-smi')
        logging.debug("Nvidia GPU detected, using CPU/GPU hybrid.")
        return True
    except Exception:
        logging.debug("No Nvidia GPU detected, using CPU only.")
        return False


def _run_case_requests(executor_cls, max_workers, case_requests, segment_name, table_name, metadata_lookup):
    with executor_cls(max_workers=max_workers) as executor:
        futures = [
            executor.submit(calc, case_request, segment_name, table_name, RESULTS_DIR, metadata_lookup)
            for case_request in case_requests
        ]
        return [task.result() for task in futures]


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('-f', '--file', type=str, help='The input CSV filename')
        group.add_argument('-p', '--protein', type=str, help='The name of the protein')
        parser.add_argument('--threads-per-worker', type=int, default=8, help='Set the number of worker threads used inside each ActSeekN worker process')
        parser.add_argument('--cpu-only', action='store_true', help='Disable GPU usage even if a GPU is available')
        args = parser.parse_args()
    except Exception as e:
        parser.print_usage()
    if args.threads_per_worker <= 0:
        raise ValueError("--threads-per-worker must be a positive integer.")
    ActSeekLib.set_thread_pool_size(args.threads_per_worker)
    ActSeekLib.set_gpu_enabled(not args.cpu_only)
    if args.file:
        input_file = Path(args.file).expanduser()
        case_requests, input_table, skipped_row_indices = _build_case_requests(input_file)
        total_requested = len(input_table)
    elif args.protein:
        case_requests = [{
            "row_index": 0,
            "query_label": _sanitize_filename_part(args.protein),
            "requested_entry": args.protein,
        }]
        total_requested = 1
        input_table = None
        skipped_row_indices = []

    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    check_gpu_availability()

    database_path = _resolve_database_path()
    if database_path.suffix == ".h5":
        distances = _read_database_entries(database_path)
    else:
        with open(database_path, 'rb') as handle:
            distances = pickle.load(handle)
    logging.debug(f"Loaded annotation database from {database_path}.")
    metadata_lookup = _build_metadata_lookup(distances)
    
    seed_protein_entries = []
    for data in distances:
        try:
            aa_cavity = {k: aa_codes[v] for k, v in data.aa_cavity.items() if v in aa_codes}
            seed_protein_entries += [(data.active_indices, data.protein_id, data.ec_number, data.seed_coords, data.cavity_coords, data.cavity_coords_cb, aa_cavity, data.active, data.real_index_seed)]
        except Exception as e:
            logging.error(f"{data.protein_id} failed with {e}.")
    logging.debug("Distances loaded.")

    try:
        table_name = "seed_protein_entries"
        segment_name = ActSeekLib.createSharedEntries(seed_protein_entries, table_name, os.getpid())
        logging.debug(f"Shared memory '{segment_name}' created.")
    except:
        logging.error("Creating shared memory failed!")
        sys.exit(1)
        
    try:
        threads_per_worker = max(1, int(ActSeekLib.get_thread_pool_size()))
        if os.getenv('SLURM_CPUS_PER_TASK'):
            MAX_PROCS = max(1, int(os.getenv('SLURM_CPUS_PER_TASK')) // threads_per_worker)
        else:
            MAX_PROCS = max(1, (os.cpu_count() or 1) // threads_per_worker)
        try:
            results = _run_case_requests(
                concurrent.futures.ProcessPoolExecutor,
                MAX_PROCS,
                case_requests,
                segment_name,
                table_name,
                metadata_lookup,
            )
        except Exception as exc:
            logging.warning(
                f"ProcessPoolExecutor failed ({exc}); falling back to ThreadPoolExecutor."
            )
            results = _run_case_requests(
                concurrent.futures.ThreadPoolExecutor,
                1,
                case_requests,
                segment_name,
                table_name,
                metadata_lookup,
            )

        success = 0
        failed_row_indices = list(skipped_row_indices)
        for case_request, result in zip(case_requests, results):
            if result["success"]:
                success += 1
            else:
                failed_row_indices.append(case_request["row_index"])
        failed = total_requested - success
        logging.debug(f'{success}/{total_requested} tasks completed, {failed}/{total_requested} tasks failed.')
        if input_table is not None:
            _write_failed_input_rows(input_file, input_table, failed_row_indices)
    except Exception as exc:
        logging.error(f"Task execution failed: {exc}")
    
    finally:
        ActSeekLib.destroySharedEntries(segment_name)
