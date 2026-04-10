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
from difflib import SequenceMatcher
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
HMMER_SEARCH_URL = "https://www.ebi.ac.uk/Tools/hmmer/api/v1/search/phmmer"
HMMER_RESULT_URL = "https://www.ebi.ac.uk/Tools/hmmer/api/v1/result/{job_id}"
ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

aa_codes = {
    "ALA": 1, "ARG": 2, "ASN": 3, "ASP": 4, "CYS": 5,
    "GLN": 6, "GLU": 7, "GLY": 8, "HIS": 9, "ILE": 10,
    "LEU": 11, "LYS": 12, "MET": 13, "PHE": 14, "PRO": 15,
    "SER": 16, "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20,
    "SEC": 21
    }

_uniprot_sequence_cache = {}


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


def _normalize_ec_number(value):
    text = _clean_optional_text(value)
    if not text:
        return None
    match = re.search(r"\d+\.\d+\.\d+\.(?:\d+|-)", text)
    return match.group(0) if match else text


def _build_result_filename(case_request, resolved_entry):
    requested_entry = case_request.get("requested_entry")
    entry_seq_ref = case_request.get("entry_seq_ref")
    original_label = _sanitize_filename_part(
        requested_entry or entry_seq_ref or case_request["query_label"]
    )
    resolved_label = _sanitize_filename_part(resolved_entry)

    if original_label == resolved_label:
        return f"actseekn-{original_label}-results.csv"

    return f"actseekn-{original_label}-inferred-{resolved_label}-results.csv"


def _find_existing_result(case_request, results_dir):
    original_label = _sanitize_filename_part(
        case_request.get("requested_entry") or case_request.get("entry_seq_ref") or case_request["query_label"]
    )

    direct_result = results_dir / f"actseekn-{original_label}-results.csv"
    if direct_result.is_file():
        return direct_result, original_label, None

    fallback_results = sorted(results_dir.glob(f"actseekn-{original_label}-inferred-*-results.csv"))
    if fallback_results:
        result_file = fallback_results[0]
        prefix = f"actseekn-{original_label}-inferred-"
        suffix = "-results.csv"
        filename = result_file.name
        if filename.startswith(prefix) and filename.endswith(suffix):
            resolved_entry = filename[len(prefix):-len(suffix)]
        else:
            resolved_entry = None
        return result_file, resolved_entry, "cached_fallback"

    return None, None, None


def _build_case_result(case_request, success, resolved_uniprot=None, inferring_method=None):
    return {
        "success": success,
        "row_index": case_request["row_index"],
        "resolved_uniprot": resolved_uniprot,
        "original_uniprot": case_request.get("requested_entry"),
        "entry_seq_ref": case_request.get("entry_seq_ref"),
        "inferring_method": inferring_method,
    }


def _wait_for_hmmer_result(job_id, timeout_seconds=600, poll_seconds=3):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(
            HMMER_RESULT_URL.format(job_id=job_id),
            headers={"Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "SUCCESS":
            return payload
        if status in {"FAILURE", "ERROR"}:
            raise RuntimeError(f"HMMER search failed with status {status}.")
        time.sleep(poll_seconds)
    raise TimeoutError(f"HMMER search {job_id} did not finish within {timeout_seconds}s.")


def _search_uniprot_by_sequence(sequence):
    fasta = f">query\n{sequence}"
    response = requests.post(
        HMMER_SEARCH_URL,
        json={"database": "uniprot", "input": fasta},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    job_id = response.json()["id"]
    result_payload = _wait_for_hmmer_result(job_id)
    hits = result_payload.get("result", {}).get("hits", [])
    candidates = []
    for hit in hits:
        metadata = hit.get("metadata") or {}
        accession = metadata.get("uniprot_accession") or metadata.get("accession")
        if not accession:
            continue
        candidates.append(
            {
                "accession": accession,
                "identifier": metadata.get("uniprot_identifier") or metadata.get("identifier"),
                "description": metadata.get("description"),
                "evalue": hit.get("evalue"),
            }
        )
    if not candidates:
        raise LookupError("Sequence search returned no UniProt hits.")
    return candidates


def _search_uniprot_by_seq_ref(entry_seq_ref, ec_number=None, size=50):
    if not entry_seq_ref:
        return []

    seq_ref = entry_seq_ref.strip()
    query_variants = [
        f'xref:RefSeq-{seq_ref}',
        f'"{seq_ref}"',
    ]
    if ec_number:
        query_variants = [f"({query}) AND (ec:{ec_number})" for query in query_variants] + query_variants

    for query in query_variants:
        response = requests.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "size": size,
                "fields": "accession,id,protein_name,ec",
            },
            timeout=60,
        )
        if response.status_code == 400:
            logging.warning(f"UniProt search rejected query '{query}' for {seq_ref}; trying next strategy.")
            continue
        response.raise_for_status()
        results = response.json().get("results", [])
        candidates = []
        for item in results:
            accession = item.get("primaryAccession")
            if not accession:
                continue
            candidates.append(
                {
                    "accession": accession,
                    "identifier": item.get("uniProtkbId"),
                    "description": (
                        ((item.get("proteinDescription") or {}).get("recommendedName") or {})
                        .get("fullName", {})
                        .get("value")
                    ),
                }
            )
        if candidates:
            logging.info(
                f"Resolved {seq_ref} to UniProt candidates via UniProt search"
                + (f" with EC {ec_number}" if ec_number else "")
                + f"; first hit {candidates[0]['accession']}."
            )
            return candidates

    return []


def _search_uniprot_by_ec(ec_number, size=200):
    if not ec_number:
        return []

    response = requests.get(
        UNIPROT_SEARCH_URL,
        params={
            "query": f"ec:{ec_number}",
            "format": "json",
            "size": size,
            "fields": "accession,id,protein_name,ec",
        },
        timeout=60,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    candidates = []
    for item in results:
        accession = item.get("primaryAccession")
        if not accession:
            continue
        candidates.append(
            {
                "accession": accession,
                "identifier": item.get("uniProtkbId"),
                "description": (
                    ((item.get("proteinDescription") or {}).get("recommendedName") or {})
                    .get("fullName", {})
                    .get("value")
                ),
            }
        )
    return candidates


def _fetch_uniprot_sequence(accession):
    if accession in _uniprot_sequence_cache:
        return _uniprot_sequence_cache[accession]

    response = requests.get(UNIPROT_ENTRY_URL.format(accession=accession), timeout=60)
    if response.status_code == 404:
        _uniprot_sequence_cache[accession] = None
        return None
    response.raise_for_status()
    sequence = ((response.json().get("sequence") or {}).get("value") or "").strip() or None
    _uniprot_sequence_cache[accession] = sequence
    return sequence


def _score_sequence_similarity(query_sequence, candidate_sequence):
    if not query_sequence or not candidate_sequence:
        return -1.0
    return SequenceMatcher(None, query_sequence, candidate_sequence).ratio()


def _rank_candidates_by_sequence(query_sequence, candidates, limit=50):
    ranked = []
    for candidate in candidates:
        accession = candidate["accession"]
        candidate_sequence = _fetch_uniprot_sequence(accession)
        similarity = _score_sequence_similarity(query_sequence, candidate_sequence)
        if similarity < 0:
            continue
        ranked.append((similarity, accession))

    ranked.sort(key=lambda item: item[0], reverse=True)
    accessions = []
    for _, accession in ranked:
        if accession not in accessions:
            accessions.append(accession)
        if len(accessions) >= limit:
            break
    return accessions


def _dedupe_accessions(accessions, exclude_accession=None, limit=50):
    deduped_accessions = []
    for accession in accessions:
        if accession == exclude_accession or accession in deduped_accessions:
            continue
        deduped_accessions.append(accession)
        if len(deduped_accessions) >= limit:
            break
    return deduped_accessions


def _infer_uniprot_fallback_tiers(case_request, exclude_accession=None):
    entry_seq_ref = case_request.get("entry_seq_ref")
    ec_number = case_request.get("ec_number")
    sequence = case_request.get("sequence")
    tiers = []

    candidates = _search_uniprot_by_seq_ref(entry_seq_ref, ec_number=ec_number)
    if candidates:
        accessions = _dedupe_accessions(
            [candidate["accession"] for candidate in candidates],
            exclude_accession=exclude_accession,
        )
        if accessions:
            tiers.append(("seq_ref", accessions))

    if ec_number and sequence:
        ec_candidates = _search_uniprot_by_ec(ec_number)
        if ec_candidates:
            accessions = _rank_candidates_by_sequence(sequence, ec_candidates)
            if accessions:
                logging.info(
                    f"{case_request['query_label']} ranked UniProt candidates within EC {ec_number}; first hit {accessions[0]}."
                )
                accessions = _dedupe_accessions(accessions, exclude_accession=exclude_accession)
                if accessions:
                    tiers.append(("ec_sequence", accessions))

    if sequence and not tiers:
        logging.info(
            f"{case_request['query_label']} could not be resolved from sequence reference"
            + (f" {entry_seq_ref}" if entry_seq_ref else "")
            + " or EC-filtered sequence ranking; falling back to HMMER sequence search."
        )
        candidates = _search_uniprot_by_sequence(sequence)
        accessions = _dedupe_accessions(
            [candidate["accession"] for candidate in candidates],
            exclude_accession=exclude_accession,
        )
        if accessions:
            tiers.append(("hmmer", accessions))

    return tiers


def _infer_uniprot_fallbacks(case_request, exclude_accession=None):
    tiers = _infer_uniprot_fallback_tiers(case_request, exclude_accession=exclude_accession)
    for method, accessions in tiers:
        try:
            resolved_entry, structure_path = _resolve_structure_from_accessions(accessions)
            return resolved_entry, structure_path, method
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No AlphaFold structure found for any fallback UniProt accession.")


def _build_case_requests(input_file):
    table = pd.read_csv(input_file)
    if "Entry" not in table.columns:
        raise ValueError(f"Input file {input_file} must contain an 'Entry' column.")

    case_requests = []
    skipped_row_indices = []
    skipped_rows = 0

    for row_index, row in table.iterrows():
        entry = _clean_optional_text(row.get("Entry"))
        sequence = _clean_optional_text(row.get("Sequence"))
        entry_seq_ref = _clean_optional_text(row.get("Entry_seq_ref"))
        ec_number = _normalize_ec_number(row.get("EC") if "EC" in row.index else row.get("EC number"))
        label = _sanitize_filename_part(entry or entry_seq_ref or f"row{row_index + 1}")
        if not entry and not sequence:
            skipped_rows += 1
            skipped_row_indices.append(row_index)
            logging.warning(f"Skipping row {row_index + 1}: empty Entry and no Sequence provided.")
            continue

        case_request = {
            "row_index": row_index,
            "query_label": label,
            "requested_entry": entry,
            "requested_accessions": [entry] if entry else [],
            "entry_seq_ref": entry_seq_ref,
            "sequence": sequence,
            "ec_number": ec_number,
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


def _write_fallback_rows(input_file, fallback_rows):
    output_path = RESULTS_DIR / f"{input_file.stem}_fallbacks.csv"
    report = pd.DataFrame(fallback_rows)
    report.to_csv(output_path, index=False)
    logging.info(f"Wrote {len(report)} fallback resolution rows to {output_path}.")


def _download_structure(accession):
    structure_path = STRUCTURES_DIR / f"AF-{accession}-F1-model_v6.pdb"
    if structure_path.is_file():
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


def _resolve_structure_from_accessions(accessions):
    for accession in accessions:
        try:
            structure_path = _download_structure(accession)
        except Exception as exc:
            logging.warning(f"{accession} structure download failed: {exc}")
            continue
        if structure_path is not None:
            return accession, structure_path
    raise FileNotFoundError("No AlphaFold structure found for any fallback UniProt accession.")


def _resolve_case_structure(case_request):
    sequence = case_request.get("sequence")
    requested_entry = case_request.get("requested_entry")
    entry_seq_ref = case_request.get("entry_seq_ref")

    if not requested_entry:
        if not sequence and not entry_seq_ref:
            raise FileNotFoundError("No UniProt entry, sequence reference, or Sequence available for inference.")
        logging.info(
            f"{case_request['query_label']} has no UniProt entry; trying inferred UniProt resolution"
            + (f" using {entry_seq_ref} as sequence reference" if entry_seq_ref else "")
            + (f" and EC {case_request['ec_number']}" if case_request.get("ec_number") else "")
            + "."
        )
        resolved_entry, structure_path, method = _infer_uniprot_fallbacks(case_request)
        return resolved_entry, structure_path, method or "sequence_inferred"

    try:
        resolved_entry, structure_path = _resolve_structure_from_accessions(case_request["requested_accessions"])
        return resolved_entry, structure_path, None
    except FileNotFoundError:
        if not requested_entry or not sequence:
            raise

        logging.warning(
            f"{case_request['query_label']} could not use UniProt entry {requested_entry}; trying inferred UniProt resolution."
        )
        resolved_entry, structure_path, method = _infer_uniprot_fallbacks(
            case_request, exclude_accession=requested_entry
        )
        return resolved_entry, structure_path, method or "sequence_inferred"


def read_pdbs(case_protein):
    parser = PDBParser()
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

        existing_result, cached_resolved_entry, cached_method = _find_existing_result(case_request, results_dir)
        if existing_result is not None:
            logging.debug(f'{query_label} had existing results.')
            return _build_case_result(
                case_request,
                True,
                cached_resolved_entry or requested_entry,
                cached_method,
            )

        resolved_entry, case_protein_filepath, resolution_method = _resolve_case_structure(case_request)
        result_file = results_dir / _build_result_filename(case_request, resolved_entry)
        if result_file.is_file():
            logging.debug(f'{query_label} had existing results.')
            return _build_case_result(case_request, True, resolved_entry, resolution_method)
        if requested_entry and requested_entry == resolved_entry:
            logging.debug(f"{query_label} used UniProt entry {resolved_entry}.")
        elif requested_entry:
            logging.debug(f"{query_label} used fallback UniProt entry {resolved_entry} instead of {requested_entry}.")
        else:
            logging.debug(f"{query_label} inferred UniProt entry {resolved_entry} from Sequence.")

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
        return _build_case_result(case_request, True, resolved_entry, resolution_method)
    except Exception as e:
        logging.error(f"{case_request.get('query_label', 'query')} failed with {e}")
        return _build_case_result(case_request, False, None, "unresolved")


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
        parser.add_argument('--cpu-only', action='store_true', help='Disable GPU usage even if a GPU is available')
        args = parser.parse_args()
    except Exception as e:
        parser.print_usage()
    if args.cpu_only:
        ActSeekLib.set_gpu_enabled(False)
    if args.file:
        input_file = Path(args.file).expanduser()
        case_requests, input_table, skipped_row_indices = _build_case_requests(input_file)
        total_requested = len(input_table)
    elif args.protein:
        case_requests = [{
            "row_index": 0,
            "query_label": _sanitize_filename_part(args.protein),
            "requested_entry": args.protein,
            "requested_accessions": [args.protein],
            "entry_seq_ref": None,
            "sequence": None,
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
        if os.getenv('SLURM_CPUS_PER_TASK'):
            MAX_PROCS = max(1, int(os.getenv('SLURM_CPUS_PER_TASK')) // 4)
        else:
            MAX_PROCS = max(1, os.cpu_count() // 4)
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
        fallback_rows = []
        for case_request, result in zip(case_requests, results):
            if result["success"]:
                success += 1
            else:
                failed_row_indices.append(case_request["row_index"])
            if result["inferring_method"]:
                fallback_rows.append({
                    "resolved_uniprot": result["resolved_uniprot"] or "",
                    "original_uniprot": result["original_uniprot"] or "",
                    "seq_ref": result["entry_seq_ref"] or "",
                    "inferring_method": result["inferring_method"],
                })
        failed = total_requested - success
        logging.debug(f'{success}/{total_requested} tasks completed, {failed}/{total_requested} tasks failed.')
        if input_table is not None:
            _write_failed_input_rows(input_file, input_table, failed_row_indices)
            _write_fallback_rows(input_file, fallback_rows)
    except Exception as exc:
        logging.error(f"Task execution failed: {exc}")
    
    finally:
        ActSeekLib.destroySharedEntries(segment_name)
