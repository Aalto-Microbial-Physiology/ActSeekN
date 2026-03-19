import os
import time
import logging
import argparse
import pickle
import wget
from pathlib import Path
import pandas as pd
from Bio.PDB import *
import subprocess
import concurrent.futures
from dataclasses import dataclass
import ActSeekLib


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
STRUCTURES_DIR = ROOT_DIR / "structures"
RESULTS_DIR = ROOT_DIR / "results"

aa_codes = {
    "ALA": 1, "ARG": 2, "ASN": 3, "ASP": 4, "CYS": 5,
    "GLN": 6, "GLU": 7, "GLY": 8, "HIS": 9, "ILE": 10,
    "LEU": 11, "LYS": 12, "MET": 13, "PHE": 14, "PRO": 15,
    "SER": 16, "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20,
    "SEC": 21
    }


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


def calc(case_protein, segment_name, table_name, results_dir):
    try:
        start_time = time.perf_counter()
        logging.debug(f'{case_protein} started...')
        case_protein_filename = f"AF-{case_protein}-F1-model_v6.pdb"
        result_file = results_dir / f"{case_protein}-result-exp.csv"
        if result_file.is_file():
            logging.debug(f'{case_protein} had existing results.')
            return True
        case_protein_filepath = STRUCTURES_DIR / case_protein_filename
        if not case_protein_filepath.is_file():
            wget.download(f"https://alphafold.ebi.ac.uk/files/{case_protein_filename}", out=str(STRUCTURES_DIR))
            logging.debug(f"{case_protein} took {time.perf_counter() - start_time:.2f}s to download the PDB file.")
        else:
            logging.debug(f"{case_protein} used local PDB file.")

        case_protein_profile = [case_protein] + list(read_pdbs(str(case_protein_filepath)))

        start_time = time.perf_counter()
        results = ActSeekLib.concurrentMain(case_protein_profile, segment_name, table_name)
        dataset = pd.DataFrame(filter(None, results), columns=['ID', 'Length hit', 'Length query', 'Mapping', 'Distances', 'Av dist', 'Arround dist', 'Structure score', 'RMSDmin','Percentage', 'EC'])
        dataset = dataset.sort_values(by=['RMSDmin']).round(6)
        dataset.to_csv(result_file)
        logging.debug(f"{case_protein} took {time.perf_counter() - start_time:.2f}s to find {len(results)} results.")
        return True
    except Exception as e:
        logging.error(f"{case_protein} failed with {e}")
        return False


def check_gpu_availability():
    try:
        subprocess.check_output('nvidia-smi')
        logging.debug("Nvidia GPU detected, using CPU/GPU hybrid.")
        return True
    except Exception:
        logging.debug("No Nvidia GPU detected, using CPU only.")
        return False


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('-f', '--file', type=str, help='The input CSV filename')
        group.add_argument('-p', '--protein', type=str, help='The name of the protein')
        args = parser.parse_args()
    except Exception as e:
        parser.print_usage()
    if args.file:
        input_file = Path(args.file).expanduser()
        case_proteins = pd.read_csv(input_file)['Entry'].tolist()
    elif args.protein:
        case_proteins = [args.protein]

    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    check_gpu_availability()

    @dataclass
    class database_entry():
        ec_number: str
        seed_coords: np.array
        cavity_coords: np.array
        cavity_coords_cb: np.array
        aa_cavity: dict
        active: np.array
        real_index_seed: dict
        protein_id: str
        active_indices: np.array

    with open(DATA_DIR / "newDatabase_v3.pickle", 'rb') as handle:
        distances = pickle.load(handle)
    
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
        pool = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCS) as executor:
            for case_protein in case_proteins:
                pool.append(executor.submit(calc, case_protein, segment_name, table_name, RESULTS_DIR))
            success = 0
            for task in pool:
                if task.result():
                    success += 1
            logging.debug(f'{success}/{len(case_proteins)} tasks completed, {len(case_proteins) - success}/{len(case_proteins)} tasks failed.')
    except:
        logging.error("All tasks failed!")
    
    finally:
        ActSeekLib.destroySharedEntries(segment_name)
