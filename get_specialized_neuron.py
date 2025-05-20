import os
import re
import json
import pdb
import argparse
from collections import Counter
from typing import Set, List, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
import matplotlib.font_manager as fm
from matplotlib_venn import venn2, venn3

from utils_neuron import *

NEURON_FILE = ["activate"]
mode_value = {
    "topscore": [0.85, 0.9, 0.95],
    "global_topk": [1, 5, 10, 50, 100, 200],
    "topk": [1, 5, 10, 15, 20, 25],
}



def get_func_and_key(mode: str):

    if mode == 'topscore':
        return get_neuron_set_layer_topscore, "score_threshold"
    elif mode == 'topk':
        return get_neuron_set_layer_topk, "top_k"
    elif mode == 'global_topk':
        return get_neuron_set_global_topk, "top_k"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def get_neuron_file_path(neuron_file: str) -> str:
    return "result_activate"


def load_results(output_neuron_dir: str, model: str, country: str, prompt_no: int, neuron_file_path: str):
    results_us_us = []
    results_other_us = []
    results_other_other = []
    results_us_other = []

    with open(f'{output_neuron_dir}/{model}-{country}_UK_{prompt_no}_{neuron_file_path}.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            results_other_us.append(json.loads(line.strip()))

    with open(f'{output_neuron_dir}/{model}-US_US_{prompt_no}_{neuron_file_path}.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            results_us_us.append(json.loads(line.strip()))
    
    with open(f'{output_neuron_dir}/{model}-{country}_{country}_{prompt_no}_{neuron_file_path}.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            results_other_other.append(json.loads(line.strip()))

    with open(f'{output_neuron_dir}/{model}-US_{country}_{prompt_no}_{neuron_file_path}.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            results_us_other.append(json.loads(line.strip()))

    return results_us_us, results_other_us, results_other_other, results_us_other
    

def specialized_neurons_calculation(setA: Set, setB: Set) -> float:
    """specialized neurons proportion"""
    if not setA and not setB:
        return 1.0
    inter = setA.intersection(setB)

    return (len(setB) - len(inter)) / len(setB)

def get_neuron_display_compare_pro():
        
    func = None
    func_key = None
    func_args = None
    output_neuron_dir = f'get_Neuron_result/{args.model}/'
    neuron_file_path = get_neuron_file_path(args.neuron_file)

    results_us_us, results_other_us, results_other_other, results_us_other = load_results(
                output_neuron_dir, args.model, args.country, args.prompt_no, neuron_file_path)


    if args.mode == 'topscore':
        func = get_neuron_set_layer_topscore
        func_args = {'score_threshold': args.score_threshold}
    elif args.mode == 'topk':
        func = get_neuron_set_layer_topk
        func_args = {'top_k': args.top_k}
    elif args.mode == 'global_topk':
        func = get_neuron_set_global_topk
        func_args = {'top_k': args.top_k}
    else:
        raise ValueError("Unsupported mode")


    if args.neuron_file == "grad_filted":
        func_args["ignore_layer"] = [31]

    func_args.update({
        'file_threshold': args.file_threshold,
        'union': True
    })

    def get_split_set(results, real_annotation, inter_ids, unique_ids):
        inter_result = []
        result_neron_id = 0
        all_result = []
        un_neron_id = 0
        for sample, data in zip(results, real_annotation.keys()):
            data = real_annotation[data]
           
            if sample["ID"] in inter_ids:
                inter_result.append(sample)
                all_result.append(sample)
                result_neron_id += 1
            else:
                all_result.append(sample)
                un_neron_id +=1

        print(f"inter {len(inter_result)}, all {len(all_result)}")
        return inter_result, all_result
    
    
    def get_neuron_dis(results_us_us,
                  results_us_other,
                  label: str = "results_us_other"):
        
        """specialized neurons calculation for Pij (results_us_other - results_us_us)"""

        if len(results_us_us) != len(results_us_other):
            raise ValueError("results_us_us and results_us_other must contain the same number of cases.")
        
        num_cases = len(results_us_us)
        print(f"Comparing {num_cases} cases...")
        
        # Initialize dictionary to collect differences per layer
        layer_diffs = {}
        
        case_avg_jaccards = []
        # Analyze each case
        for case_idx, (case_us, case_other) in enumerate(zip(results_us_us, results_us_other), 1):
            # Group neurons by layer for each result set
            layers_us = {}
            for layer, neuron_id in case_us:
                layers_us.setdefault(layer, set()).add(neuron_id)
            
            layers_other = {}
            for layer, neuron_id in case_other:
                layers_other.setdefault(layer, set()).add(neuron_id)
            
            # Get all layers involved in this case
            all_layers = set(layers_us.keys()).union(layers_other.keys())
            case_jaccards = []
            
            for layer in all_layers:
                neurons_us = layers_us.get(layer, set())
                neurons_other = layers_other.get(layer, set())
                
                jaccard = specialized_neurons_calculation(neurons_us, neurons_other)
                case_jaccards.append(jaccard)
                
                # Calculate unique neurons
                unique_us = len(neurons_us - neurons_other)
                unique_other = len(neurons_other - neurons_us)
                
                # Record differences
                if layer not in layer_diffs:
                    layer_diffs[layer] = {"jaccard": [], "unique_us": [], "unique_other": []}
                
                layer_diffs[layer]["jaccard"].append(jaccard)
                layer_diffs[layer]["unique_us"].append(unique_us)
                layer_diffs[layer]["unique_other"].append(unique_other)

            if case_jaccards:
                avg_jaccard_case = sum(case_jaccards) / len(case_jaccards)
            else:
                avg_jaccard_case = 0
            case_avg_jaccards.append(avg_jaccard_case)
            
        
        # Check if any layers were recorded
        if not layer_diffs:
            print("No layer differences recorded. Please check your input data.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Summarize data per layer
        summary_data = []
        sum_avg_jaccard = 0.0
        sum_len = 0
        for layer, metrics in layer_diffs.items():
            sum_avg_jaccard += sum(metrics["jaccard"])
            sum_len += len(metrics["jaccard"])
            avg_jaccard = sum(metrics["jaccard"]) / len(metrics["jaccard"]) if metrics["jaccard"] else 0
            total_unique_us = sum(metrics["unique_us"])
            total_unique_other = sum(metrics["unique_other"])
            summary_data.append({
                "layer": layer,
                "avg_jaccard": avg_jaccard,
                "total_unique_us": total_unique_us,
                "total_unique_other": total_unique_other,
                "num_cases": len(metrics["jaccard"])
            })
        
        df_summary = pd.DataFrame(summary_data)

        case_avg_jaccards_series = pd.Series(case_avg_jaccards)
        stats = {
            "proportion of specialized neurons": case_avg_jaccards_series.mean()
        }
        for key, value in stats.items():
            print(f"{key}: {value:.4f}", end = "")
        print("==========================")

        if 'layer' not in df_summary.columns:
            print("Error: 'layer' column is missing from the summary DataFrame.")
            print("Available columns:", df_summary.columns.tolist())
            print("DataFrame content:")
            print(df_summary.head())
            return df_summary, pd.DataFrame()
        
        # Convert 'layer' to integer if possible for sorting
        if not pd.api.types.is_numeric_dtype(df_summary['layer']):
            try:
                df_summary['layer'] = df_summary['layer'].astype(int)
            except ValueError:
                print("Warning: 'layer' column cannot be converted to integers. Converting to strings.")
                df_summary['layer'] = df_summary['layer'].astype(str)
        
        # Sort by layer
        df_summary = df_summary.sort_values("layer")

        # Create DataFrame for specialized neurons
        jaccard_records = []
        for layer, metrics in layer_diffs.items():
            for j in metrics["jaccard"]:
                jaccard_records.append({"layer": layer, "jaccard_similarity": j})
        
        df_jaccard = pd.DataFrame(jaccard_records)
        

        return df_summary, df_jaccard, case_avg_jaccards
        

    # load  reponse for Qii, Qji, Qjj, Qij
    lang1_content1 = pd.read_csv(os.path.join("model_inference_results/",f'{args.model}/{args.model}_US_US_{args.prompt_no}_response_score.csv'),encoding='utf-8')
    lang1_content2 = pd.read_csv(os.path.join("model_inference_results/",f'{args.model}/{args.model}_{args.country}_UK_{args.prompt_no}_response_score.csv'),encoding='utf-8')
    lang2_content2 = pd.read_csv(os.path.join("model_inference_results/",f'{args.model}/{args.model}_{args.country}_{args.country}_{args.prompt_no}_response_score.csv'),encoding='utf-8')
    lang2_content1 = pd.read_csv(os.path.join("model_inference_results/",f'{args.model}/{args.model}_US_{args.country}_{args.prompt_no}_response_score.csv'),encoding='utf-8')
    filtered_lang1_content1 = lang1_content1[lang1_content1['answer_index'] != - 1]
    filtered_lang2_content1 = lang2_content1[lang2_content1['answer_index'] != - 1]
    lang1_content1_indexed = filtered_lang1_content1.set_index(['ID', 'answer_index'])
    lang2_content1_indexed = filtered_lang2_content1.set_index(['ID', 'answer_index'])
    matching_rows_count_q1 = lang1_content1_indexed.index.intersection(lang2_content1_indexed.index)

    # remove q that is not answerable in BLEnD
    filtered_lang1_content2 = lang1_content2[lang1_content2['answer_index'] != -1]
    filtered_lang2_content2 = lang2_content2[lang2_content2['answer_index'] != -1]
    lang1_content2_indexed = filtered_lang1_content2.set_index(['ID', 'answer_index'])
    lang2_content2_indexed = filtered_lang2_content2.set_index(['ID', 'answer_index'])
    matching_rows_count_q2 = lang1_content2_indexed.index.intersection(lang2_content2_indexed.index)


    inter_ids = set()
    for index in matching_rows_count_q2:
        id = index[0]  
        inter_ids.add(id)
    
    unique_ids_us_us = set()
    for index, row in lang1_content1_indexed.iterrows():
        id = index[0]
        unique_ids_us_us.add(id)
    
    unique_ids_us_other = set()
    for index, row in lang2_content1_indexed.iterrows():
        id = index[0]
        unique_ids_us_other.add(id)
    
    unique_ids_other_other = set()
    for index, row in lang2_content2_indexed.iterrows():
        id = index[0]
        unique_ids_other_other.add(id)

    unique_ids_other_us = set()
    for index, row in lang1_content2_indexed.iterrows():
        id = index[0]
        unique_ids_other_us.add(id)
    
    inter_ids_us = set()
    for index in matching_rows_count_q1:
        id = index[0]  
        inter_ids_us.add(id)

    with open(os.path.join("data/annotations",f"{args.country}_data.json"),'r') as f:
        real_annotation = json.load(f)
    
    with open(os.path.join("data/annotations",f"US_data.json"),'r') as f:
        real_annotation_us = json.load(f)

    
    result_inter_us_us, result_inter_all = get_split_set(results_us_us, real_annotation_us, inter_ids_us, unique_ids_us_us)
    result_inter_us_other, result_inter_all_us = get_split_set(results_us_other, real_annotation, inter_ids_us, unique_ids_us_other)
    results_us_us_list = [] 
    results_us_other_list = []
    for sample_us_us, sample_us_other in zip(result_inter_all, result_inter_all_us):
        results_us_us_list.append(list(func([sample_us_us], **func_args)))
        results_us_other_list.append(list(func([sample_us_other], **func_args)))

    # P en_j
    _, _, jaccard = get_neuron_dis(results_us_us_list, results_us_other_list, f"results_us_{args.country}")


    result_inter_us_us, result_inter_all = get_split_set(results_other_other, real_annotation, inter_ids, unique_ids_other_other)
    result_inter_us_other, result_inter_all_us = get_split_set(results_other_us, real_annotation_us, inter_ids, unique_ids_other_us)
    results_other_other_list = []
    results_other_us_list = []
    for sample_us_us, sample_us_other in zip(result_inter_all, result_inter_all_us):
        results_other_us_list.append(list(func([sample_us_other], **func_args)))
        results_other_other_list.append(list(func([sample_us_us], **func_args)))

    # P j_j
    _, _, jaccard = get_neuron_dis(results_other_us_list, results_other_other_list, f"results_{args.country}_{args.country}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuron Selection Script")
    parser.add_argument('--model',type=str,default="Llama3-8b-Instruct",help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable. If you want to test on multiple models, provide multiple model names with ", " between each (e.g., "gpt-4-0125-preview, aya-101").')
    parser.add_argument('--country',type=str,default="China",help='',)
    parser.add_argument('--language',type=str,default="UK",help='',)
    parser.add_argument('--prompt_no',type=str,default="inst-8", help='Provide the propmt id (ex. inst-1, inst-2, pers-1, etc.')
    parser.add_argument('--top_k', type=int, default=5, help="Number of top neurons to select (used in 'topk' and 'global_topk' modes)")
    parser.add_argument('--score_threshold', type=float, default=-1, help="Score threshold multiplier (used in 'topscore' mode)")
    parser.add_argument('--file_threshold', type=float, default=1.0, help="Fraction of results to process")
    parser.add_argument('--output', type=str, default='neuron/neurons_setting.json', help="Output path for neuron settings")
    parser.add_argument('--neuron_file', type=str, default='activate')
    parser.add_argument('--neuron_file_path', type=str, default='result_activate')
    parser.add_argument('--mode', type=str, default="topk")
    args = parser.parse_args()


    for country in ["China", "Spain", "Iran", "Indonesia", "South_Korea", "West_Java"]:
        args.country = country
        get_neuron_display_compare_pro()
