"""
This file combines the weights of the two models into one by taking the average.
"""

import torch
import os
import argparse
from datetime import datetime
import collections

# Function that takes a `model.ckpt` file and loads it using PyTorch
def load_model(path: str):
    # Load the model
    model = torch.load(path, map_location="cpu")

    # Get the weights
    print(f"Model keys: {model.keys()}")
    weights = model['state_dict']

    # Return the weights
    return weights


# Function that takes the average of two weights, each the outcome of model['model_state_dict']
def average_weights(weights1: collections.OrderedDict, weights2: collections.OrderedDict):
    # Assert that both weights have the same shape
    assert weights1.keys() == weights2.keys(), "The weights do not have the same keys."
    
    # Get the keys of the weights
    keys = weights1.keys()
    # Create a dictionary to store the averaged weights
    averaged_weights = {}
    averaged_weights['state_dict'] = {}
    # Loop over the keys
    for key in keys:
        # Get the weights
        weight1 = weights1[key]
        weight2 = weights2[key]
        # Take the average
        averaged_weight = (weight1 + weight2) / 2.
        # Store the averaged weight
        averaged_weights['state_dict'][key] = averaged_weight
    # Return the averaged weights
    return averaged_weights


# Get parser function
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, required=True)
    parser.add_argument("--path2", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


# Asserts properties of arguments
def assert_args(args: argparse.Namespace):
    # A utility function that ensures that a path exists
    def _ensure_path(path: str) -> bool:
        return os.path.exists(path)
    # Assert that the output directory exists
    def _ensure_dir(path: str) -> bool:
        return os.path.isdir(path)
    # Assert that the paths exist
    assert _ensure_path(args.path1), f"Path {args.path1} does not exist."
    assert _ensure_path(args.path2), f"Path {args.path2} does not exist."
    assert _ensure_dir(args.output_dir), f"Path {args.output_dir} does not exist."
    # Assert that the paths are not the same
    assert args.path1 != args.path2, "The paths are the same."
    # Assert that all paths are ckpt files
    assert args.path1.endswith(".ckpt"), "Path 1 is not a ckpt file."
    assert args.path2.endswith(".ckpt"), "Path 2 is not a ckpt file."
    # Assert that the paths either have "waymo" or "nuimages" in them
    assert "waymo" in args.path1 or "nuimages" in args.path1, "Path 1 is not a waymo or nuimages model."
    assert "waymo" in args.path2 or "nuimages" in args.path2, "Path 2 is not a waymo or nuimages model."



if __name__ == "__main__":
    """
    Example usage:
    ```
    time python3 exploration/weights_merge.py \
    --path1 $SCRATCH/LOGS/diffusion-for-auto/nuimages/2023-10-19T15-54-56_nuimages-ldm-vq-4/checkpoints/last.ckpt \
    --path2 $SCRATCH/LOGS/diffusion-for-auto/waymo/2023-10-18T16-14-53_waymo-ldm-vq-4/checkpoints/last.ckpt \
    --output_dir $SCRATCH/LOGS/diffusion-for-auto/joint
    ```
    """
    # Get paths of both models through parser
    args = get_parser().parse_args()
    assert_args(args)
    
    # Loads two models
    weights1 = load_model(args.path1)
    weights2 = load_model(args.path2)
    
    # Takes the average of the weights
    averaged_weights = average_weights(weights1, weights2)
    # Saves the averaged weights; add joint file name path1 + path2 + datetime to outputpath
    data_1 = "waymo" if "waymo" in args.path1 else "nuimages"
    data_2 = "waymo" if "waymo" in args.path2 else "nuimages"
    torch.save(averaged_weights, os.path.join(args.output_dir,
                                              data_1 + '-' + data_2 + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".ckpt"))