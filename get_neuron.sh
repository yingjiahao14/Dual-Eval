#!/bin/bash

# List of countries
countries=("China" "Spain" "Iran" "Indonesia" "South_Korea" "West_Java")

# Loop through each country
for country in "${countries[@]}"; do
    echo "Processing for country: $country"
    
    # Run the commands for each country
    python get_neuron.py --country "$country" --language "UK"
    python get_neuron.py --country "$country" --language "$country"
    python get_neuron.py --country "US" --language "$country"
    
done

echo "All tasks completed."