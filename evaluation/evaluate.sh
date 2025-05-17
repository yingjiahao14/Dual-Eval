#!/bin/bash

# Define model keys
MODEL_KEYS=(
    "Llama3-8b-Instruct"
    "gemma-2-9b-it"
    "bloomz-7b1"
    "claude-3-5"
    "command-r"
    "gpt-4o-2024-08-06"
    "Llama3-8b-70B-Instruct"
    "Qwen2.5-7B-Instruct"
)

declare -A COUNTRY_LANG

COUNTRY_LANG["West_Java"]="West_Java,UK"
COUNTRY_LANG["China"]="China,UK"
COUNTRY_LANG["US"]="US,China,South_Korea,Indonesia,Spain,West_Java,Iran"
COUNTRY_LANG["South_Korea"]="South_Korea,UK"
COUNTRY_LANG["Indonesia"]="Indonesia,UK"
COUNTRY_LANG["Spain"]="Spain,UK"
COUNTRY_LANG["Iran"]="Iran,UK"

PROMPT_NUMBERS=("inst-8")
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda-devices)
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --model-keys)
            IFS=',' read -ra MODEL_KEYS <<< "$2"
            shift 2
            ;;
        --country-lang)
            IFS=',' read -ra country_lang_pairs <<< "$2"
            for pair in "${country_lang_pairs[@]}"; do
                IFS=':' read -ra kv <<< "$pair"
                COUNTRY_LANG["${kv[0]}"]="${kv[1]}"
            done
            shift 2
            ;;
        --prompt-numbers)
            IFS=',' read -ra PROMPT_NUMBERS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

for country in "${!COUNTRY_LANG[@]}"; do
    echo "$country: ${COUNTRY_LANG[$country]}"
done

# PROMPT_NUMBERS=("inst-4" "pers-3")

# Iterate over models, countries, languages, and prompts
for model_key in "${MODEL_KEYS[@]}"; do
    for country in "${!COUNTRY_LANG[@]}"; do
        IFS=',' read -ra languages <<< "${COUNTRY_LANG[$country]}"
        for language in "${languages[@]}"; do
            for prompt_no in "${PROMPT_NUMBERS[@]}"; do
                echo $model_key
                python evaluate.py --model "$model_key" \
                                    --language "$language" \
                                    --country "$country" \
                                    --prompt_no "$prompt_no" \
                                    --id_col ID \
                                    --question_col Translation \
                                    --response_col response \
                                    --annotation_filename "${country}_data.json" \
                                    --annotations_key "annotations" \
                                    --evaluation_result_file "evaluation_results.csv"
            done
        done
    done
done




