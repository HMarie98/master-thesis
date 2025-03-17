import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import itertools
import os

generator = np.random.default_rng(18)
    
def random_true_scores(L):
    '''
    Randomly generate the ground truth scores of objects.

    Args:
        L (int): Number of objects.

    Returns:
        dict: Maps object to score
    '''
    scores = [int(x) for x in generator.choice(range(1, L+1), L, replace=False)]
    return dict(zip(range(L),scores))

# Parameters
L = 60 # Number of objects
W = 100 # Number of workers
N_w = 100 # Task per worker
k = 3 # k-ary subsets

true_scores = random_true_scores(L)

def sample_k_ary_subsets(task_generator, objects = L, k=3, n_samples=N_w):
    '''
    Randomly sample k-ary subsets of objects

    Args:
        L (int): Number of objects
        k (int): Subset size
        n_samples (int): Number of samples

    Returns:
        np.ndarray: Array of k-ary subsets sampled from the set of objects
    '''
    gen =  np.random.default_rng(task_generator)
    subsets = []

    while len(subsets) < n_samples:
        sample = [int (x) for x in sorted(gen.choice(L, k, replace=False))]
        if sample not in subsets:
            subsets.append(sample)

    return subsets

# Dirichlet priors
alpha_0_expert = [85, 10, 4, 1]
alpha_0_amateur = [30, 60, 8, 2]
alpha_0_spammer = [25, 25, 25, 25]
alpha_0_malicious = [1, 3, 6, 90]

def get_worker_qualities(setting):
    '''
    Samples worker qualities from a dirichlet distribution based on the worker type.

    Args:
        setting (str): The type of worker quality to sample.
            Valid options are: 'expert', 'amateur', 'spammer', and 'malicious'.

    Returns:
        numpy.ndarray: A matrix of sampled worker qualities. Each column corresponds to a worker. Each row is the sampled Dirichlet vector based on the `alpha_0` parameter of the given worker type.
    
    Raises:
        ValueError: If the input `setting` is not one of the allowed worker types.
    '''
    match setting:
        case 'expert':
            return generator.dirichlet(alpha_0_expert, size=W)
        case 'amateur':
            return generator.dirichlet(alpha_0_amateur, size=W)
        case 'spammer':
            return generator.dirichlet(alpha_0_spammer, size=W)
        case 'malicious':
            return generator.dirichlet(alpha_0_malicious, size=W)
        case _:
            raise ValueError('Enter expert, amateur, spammer, or malicious.')
        
def kendall_tau_distance(rank_a, rank_b):
    '''
    Compute the Kendall tau distance between two rankings.
    
    Args:
        rank_a (list): First ranking.
        rank_b (list): Second ranking.

    Returns:
        float: Kendall tau distance between rank_a and rank_b.
    '''
    tau, _ = kendalltau(rank_a, rank_b)
    n = len(rank_a)
    total_pairs = n * (n-1) // 2
    return (1 - tau) * total_pairs

def ranking_with_kendall_tau(true_ranking, target_distance):
    '''
    Generate a ranking with a Kendall tau distance close to the target distance.

    Args:
        true_ranking (list): True ranking.
        target_distance (int): Desired Kendall tau distance.

    Returns:
        list: _description_
    '''
    
    all_rankings = list(itertools.permutations(true_ranking))
    kt_dist = []
    for rank in all_rankings:
        kt_dist.append(kendall_tau_distance(true_ranking, list(rank)))
    valid_rankings = [ranking for ranking in all_rankings if kendall_tau_distance(true_ranking, list(ranking)) == target_distance*2]

    if valid_rankings:
        return list(int(x) for x in generator.choice(valid_rankings))
        
def generate_worker_labels(setting):
    '''
    Generate worker labels

    Args:
        setting (str): Expert, amateur, spammer or malicious

    Returns:
        list: labeled subsets
    '''
    worker_qualities = get_worker_qualities(setting)
    labels = []
    for w in range(W):
        eta_w = worker_qualities[w]
        tasks = sample_k_ary_subsets(w)
        for task in tasks:
            
            #subset= subsets[task]
            subset_scores = [true_scores[key] for key in task]
            true_rank = np.array(sorted(range(len(task)), key=lambda i: subset_scores[i], reverse=True))
            sorted_subset = [task[i] for i in true_rank]

            subset_distance = generator.choice([0,1,2,3], p=eta_w)
            noisy_sorted = ranking_with_kendall_tau(sorted_subset, subset_distance)

            labels.append({'worker id': w, 'subset': task, 'true rank': sorted_subset, 'noisy rank': noisy_sorted})
     
    return pd.DataFrame(labels)

expert_df = generate_worker_labels('expert')
amateur_df = generate_worker_labels('amateur')
spammer_df = generate_worker_labels('spammer')
malicious_df = generate_worker_labels('malicious')

expert_df.to_csv('datasets/expert.csv', index=False)
amateur_df.to_csv('datasets/amateur.csv', index=False)
spammer_df.to_csv('datasets/spammer.csv', index=False)
malicious_df.to_csv('datasets/malicious.csv', index=False)