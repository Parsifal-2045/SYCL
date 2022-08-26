import numpy as np
import pandas as pd
import os
import argparse

TRUTH_RESULTS_BASE_DIR = "/afs/cern.ch/user/j/jolivera/public/true_data/output/output"
EXPERIMENTS_BASE_DIR = ""
TOLERANCE = 0.1

def _can_be_compared(vecA, vecB):
    return vecA.size == vecB.size

def is_perfect_match(vecA, vecB):
    return _can_be_compared(vecA, vecB) and np.array_equal(vecA, vecB)

def is_tolerant_match(vecA, vecB, tolerance):
    if not _can_be_compared(vecA, vecB):
        return False

    diff = np.abs(vecA-vecB)
    return np.all(diff < tolerance)

def _get_clusterid_sorted_counts(vec):
    freq_table = {}
    for x in vec:
        if x not in freq_table:
            freq_table[x] = 0
        freq_table[x] += 1
    fr_counts = list(freq_table.values())
    fr_counts.sort()
    return fr_counts

def is_clusters_equivalent(vecA, vecB):
    if not _can_be_compared(vecA, vecB):
        return False
        
    fr_A = _get_clusterid_sorted_counts(vecA)
    fr_B = _get_clusterid_sorted_counts(vecB)

    return np.array_equal(np.array(fr_A), np.array(fr_B))

def compare_results(experiment_results_filename, truth_results_filename, tolerance):
    truth = pd.read_csv(os.path.join(TRUTH_RESULTS_BASE_DIR, truth_results_filename))
    exper = pd.read_csv(os.path.join(EXPERIMENTS_BASE_DIR, experiment_results_filename))

    is_seeds = is_perfect_match(exper.isSeed, truth.isSeed)
    if not is_seeds:
        print("Failed is_seeds comparison!")
    is_nh = is_perfect_match(exper.nh, truth.nh)
    if not is_nh:
        print("Failed is_nh comparison!")
    is_distance = is_tolerant_match(exper.delta, truth.delta, tolerance)
    if not is_distance:
        print("Failed is_distance comparison!")
    is_density = is_tolerant_match(exper.rho, truth.rho, tolerance)
    if not is_density:
        print("Failed is_density comparison!")
    is_clusters = is_clusters_equivalent(exper.clusterId, truth.clusterId)
    if not is_clusters:
        print("Failed is_clusters comparison!")

    return is_seeds and is_nh and is_distance and is_density and is_clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLUE Results Validator')
    parser.add_argument('dir', help='Path of the directory where the experiment results are located')
    parser.add_argument('experiments', nargs='+', help='Names of experiments to be validated')

    args = parser.parse_args()

    EXPERIMENTS_BASE_DIR = args.dir
    truth_experiment_table = {
        'toyDetector_1000_20.00_25.00_2.00.csv': 'toyDetector_1000_20.00_25.00_2.00.csv',
        'toyDetector_2000_20.00_25.00_2.00.csv': 'toyDetector_2000_20.00_25.00_2.00.csv',
        'toyDetector_3000_20.00_25.00_2.00.csv': 'toyDetector_3000_20.00_25.00_2.00.csv',
        'toyDetector_4000_20.00_25.00_2.00.csv': 'toyDetector_4000_20.00_25.00_2.00.csv',
        'toyDetector_5000_20.00_25.00_2.00.csv': 'toyDetector_5000_20.00_25.00_2.00.csv',
        'toyDetector_6000_20.00_25.00_2.00.csv': 'toyDetector_6000_20.00_25.00_2.00.csv',
        'toyDetector_7000_20.00_25.00_2.00.csv': 'toyDetector_7000_20.00_25.00_2.00.csv',
        'toyDetector_8000_20.00_25.00_2.00.csv': 'toyDetector_8000_20.00_25.00_2.00.csv',
        'toyDetector_9000_20.00_25.00_2.00.csv': 'toyDetector_9000_20.00_25.00_2.00.csv',
        'toyDetector_10000_20.00_25.00_2.00.csv': 'toyDetector_10000_20.00_25.00_2.00.csv',
    }

    for exper_name in args.experiments:
        print(f'Comparing {exper_name} against {truth_experiment_table[exper_name]}...')
        if compare_results(exper_name, truth_experiment_table[exper_name], TOLERANCE):
            print('Results are equal')
        else:
            print('Results are NOT EQUAL!')