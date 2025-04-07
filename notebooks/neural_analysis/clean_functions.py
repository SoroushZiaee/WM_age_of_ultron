import matplotlib.pyplot as plt
import numpy as np

import h5py # load neural data
from typing import List, Dict, Any

def load_neural_data(file_path: str, contents: List[str]):
    neural_data_dict = {}
    
    with h5py.File(file_path, 'r') as f:
        for content in contents:
            c = f[content]
            if c.shape == ():
                c_val = c[()]
            else:
                c_val = c[:]
            neural_data_dict[content] = c_val


    return neural_data_dict

def transpose_neural_data(neural_data: Dict[str, Any]):
    # First perform transpose data
    # currently we have (timebins(0), images(1), neural_sites(2), reps(3))
    # what do we want? (reps, images, neural_sites, timebins)

    for key, value in neural_data.items():
        neural_data[key] = np.transpose(value, axes=(3, 1, 2, 0))

    return neural_data

# select content
def select_contents(delay: int, task: str):
    active_cases = {
        400: "r4",
        800: "r5",
        1200: "r6"
    }
    
    passive_cases = {
        400: "r1",
        800: "r2",
        1200: "r3"
    }
    
    if task == "active":
        return active_cases.get(delay, None)
    else:
        return passive_cases.get(delay, None)

def get_percent_correct_from_proba(probabilities, true_labels, class_order=None):
    """
    Calculate the ratio of true class probability to the sum of each competing class probability.
    
    Parameters:
    -----------
    probabilities : numpy.ndarray
        Model prediction probabilities, shape (n_samples, n_classes)
    true_labels : array-like
        True class labels for each sample
    class_order : array-like, optional
        Order of the classes. If None, will be inferred from unique values in true_labels
        
    Returns:
    --------
    numpy.ndarray
        Array of shape (n_samples, n_classes) containing the ratio metrics.
        NaN values indicate the true class for each sample.
    """
    # Get number of samples
    n_samples = probabilities.shape[0]
    
    # If class_order is not provided, infer it from the labels
    if class_order is None:
        class_order = np.unique(true_labels)
    
    # Initialize output array with NaN values
    percent_correct = np.full((n_samples, len(class_order)), np.nan)
    
    # Process each sample
    for i in range(n_samples):
        # Find the true class for this sample
        true_class_mask = (true_labels[i] == class_order)
        
        # Get the probability for the true class
        true_class_prob = probabilities[i, true_class_mask]
        
        # For each class, calculate: true_prob / (true_prob + competing_prob)
        # This creates a ratio showing how strongly the model prefers the true class
        # over each individual competing class
        for j, is_true_class in enumerate(true_class_mask):
            if not is_true_class:  # Skip the true class (will remain NaN)
                competing_prob = probabilities[i, j]
                percent_correct[i, j] = true_class_prob / (true_class_prob + competing_prob)
    
    return percent_correct

# generate time for x-axis
def generate_times(init, end, bins_duration):
    return np.arange(init, end + 1, bins_duration)

# Generate beautiful plot
def make_a_plot_beautiful(ax=None):
    if ax is None:
        ax = plt.gca()

    # first let's fix ticks of our axis
    ax.tick_params(direction="out", length=5, width=1)

    # remove top and right line, and make bottom and left line thicker
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_linewidth(1)
    ax.spines["left"].set_linewidth(1)

    # create 1:1 aspect ratio, and set background to white
    ax.set_facecolor("white")
    ax.set_box_aspect(1)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = '15'

    return ax
    
    