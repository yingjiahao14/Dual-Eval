import torch
import copy
import random
import os
import argparse
from collections import Counter
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import re
import pdb
import seaborn as sns


# Collect neuron set by keeping top-k activations for each layer and sample
def get_neuron_set_layer_topk(result, top_k=100, file_threshold=1.0, union=True, ignore_layer=None):
    cutoff = int(len(result) * file_threshold)
    neuron_set_all = set()
    for item_index, item in enumerate(result[:cutoff]):
        neuron_set = set()
        neuron_list = item.get("top_neurons", [])
        scores_list = {}
        for neuron in neuron_list:
            score_key = "score" if "score" in neuron else "attribution_score"
            layer = neuron.get("layer")
            neuron_idx = neuron.get("neuron")
            score = neuron.get(score_key, 0)
            if layer is None or neuron_idx is None:
                continue 
            if ignore_layer is not None and int(layer) in ignore_layer:
                continue  
            if layer not in scores_list:
                scores_list[layer] = []
            scores_list[layer].append({"neuron": neuron_idx, "score": score})
        for layer_number, neurons in scores_list.items():
            if not neurons:
                continue  
            neurons_sorted = sorted(neurons, key=lambda x: x["score"], reverse=True)
            # Keep only top_k
            top_neurons = neurons_sorted[:top_k]
            for neuron in top_neurons:
                neuron_set.add((layer_number, neuron["neuron"]))
        if not neuron_set_all:
            neuron_set_all = neuron_set
        else:
            neuron_set_all = neuron_set_all | neuron_set if union else neuron_set_all & neuron_set
    return neuron_set_all

# Collect a neuron set based on per-layer score thresholding in each file
def get_neuron_set_layer_topscore(result, score_threshold=-1, file_threshold=1.0, union=True, ignore_layer=None):
    cutoff = int(len(result) * file_threshold)
    neuron_set_all = set()
    for item in result[:cutoff]:
        neuron_set = set()
        neuron_list = item.get("top_neurons", [])
        scores_list = {} 
        for neuron in neuron_list:
            score_key = "score" if "score" in neuron else "attribution_score"
            layer = neuron.get("layer")
            neuron_idx = neuron.get("neuron")
            score = neuron.get(score_key, 0)
            if layer is None or neuron_idx is None:
                continue  
            if ignore_layer is not None and int(layer) in ignore_layer:
                continue  
            if layer not in scores_list:
                scores_list[layer] = []
            scores_list[layer].append({"neuron": neuron_idx, "score": score})
        for layer_number, neurons in scores_list.items():
            if not neurons:
                continue  
            # Sort neurons by activation score
            neurons_sorted = sorted(neurons, key=lambda x: x["score"], reverse=True)
            # If threshold set, only keep neurons above a certain proportion of the top score
            if score_threshold != -1:
                top_score = neurons_sorted[0]["score"]
                threshold_score = top_score * score_threshold
            else:
                threshold_score = None
            for neuron in neurons_sorted:
                if score_threshold != -1:
                    if neuron["score"] >= threshold_score:
                        neuron_set.add((layer_number, neuron["neuron"]))
                    else:
                        break  
                else:
                    neuron_set.add((layer_number, neuron["neuron"]))
        # Union or intersection across all files/samples
        if not neuron_set_all:
            neuron_set_all = neuron_set
        else:
            neuron_set_all = neuron_set_all | neuron_set if union else neuron_set_all & neuron_set
    return neuron_set_all 

# Collect neuron set by keeping global top-k activations per sample (not per layer)
def get_neuron_set_global_topk(result, top_k=100, file_threshold=1.0, union=True, ignore_layer=None):
    cutoff = int(len(result) * file_threshold)
    neuron_set_all = set()
    for item_index, item in enumerate(result[:cutoff]):
        neuron_set = set()
        neuron_list = item.get("top_neurons", [])
        all_neurons = []
        for neuron in neuron_list:
            score_key = "score" if "score" in neuron else "attribution_score"
            layer = neuron.get("layer")
            neuron_idx = neuron.get("neuron")
            score = neuron.get(score_key, 0)
            if layer is None or neuron_idx is None:
                continue  
            if ignore_layer is not None and int(layer) in ignore_layer:
                continue  
            all_neurons.append({"layer": layer, "neuron": neuron_idx, "score": score})
        # Sort all neurons (all layers mixed)
        all_neurons_sorted = sorted(all_neurons, key=lambda x: x["score"], reverse=True)
        top_neurons = all_neurons_sorted[:top_k]
        for neuron in top_neurons:
            neuron_set.add((neuron["layer"], neuron["neuron"]))
        if not neuron_set_all:
            neuron_set_all = neuron_set
        else:
            neuron_set_all = neuron_set_all | neuron_set if union else neuron_set_all & neuron_set
    return neuron_set_all

# Given a neuron set, zero these neurons' weights/bias in one model copy,
# and also zero random neurons in another copy for control comparison.
def zero_neurons_and_random(model, neuron_set, random_seed=None):
    model_zero_specified = copy.deepcopy(model)
    model_zero_random = copy.deepcopy(model)
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
    all_neurons = []
    num_layers = len(model.model.layers)  
    for layer_idx in range(num_layers):
        up_proj = model.model.layers[layer_idx].mlp.up_proj
        num_neurons = up_proj.weight.size(0)
        for neuron_idx in range(num_neurons):
            all_neurons.append((layer_idx, neuron_idx))
    num_zero = len(neuron_set)
    available_neurons = list(set(all_neurons))
    if num_zero > len(available_neurons):
        raise ValueError("Not enough neurons to sample from.")
    random_neurons = random.sample(available_neurons, num_zero)
    def zero_specific_neurons(model_copy, neurons):
        for layer, neuron in neurons:
            try:
                up_proj = model_copy.model.layers[layer].mlp.up_proj
                up_proj.weight.data[neuron].zero_()
                if up_proj.bias is not None:
                    up_proj.bias.data[neuron].zero_()
            except (IndexError, AttributeError):
                print(f"Invalid layer {layer} or neuron {neuron}.")
    zero_specific_neurons(model_zero_specified, neuron_set)
    zero_specific_neurons(model_zero_random, random_neurons)
    return model_zero_specified, model_zero_random