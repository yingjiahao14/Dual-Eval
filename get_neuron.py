from utils import *
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Collate function for DataLoader: Pads input sequences to the same length
def create_collate_fn(tokenizer):
    pad_token_id = tokenizer.pad_token_id  

    def collate_fn(batch):
        # Pad input_ids and attention_mask
        input_ids = pad_sequence([item['input_ids'].squeeze() for item in batch], batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence([item['attention_mask'].squeeze() for item in batch], batch_first=True, padding_value=0)
        # Convert other fields into tensor or list as appropriate
        prompt_indices = torch.tensor([item['prompt_index'] for item in batch])
        prompt_end_indices = torch.tensor([item['prompt_end_index'] for item in batch])
        IDs = [item['ID'] for item in batch]
        prompts_text = [item['prompt_text'] for item in batch]
        responses_text = [item['response_text'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prompt_indices': prompt_indices,
            'prompt_end_indices': prompt_end_indices,
            'IDs': IDs,
            'prompts_text': prompts_text,
            'responses_text': responses_text
        }

    return collate_fn

# Custom Dataset: Handles the logic for getting one item (sample) for DataLoader
class ResponseDataset(Dataset):
    def __init__(self, dataframe, tokenizer, model):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        question = row['prompt']
        response = row['response']
        # Format the prompt and response according to model type
        prompt, prompt_index, prompt_end_index = get_formated_instruction(question, response, self.model, self.tokenizer)

        return {
            'input_ids': prompt['input_ids'],
            'attention_mask': prompt['attention_mask'],
            'prompt_index': prompt_index,
            'prompt_end_index': prompt_end_index,
            'ID': row['ID'],
            'prompt_text': row['prompt'],
            'response_text': row['response']
        }

# Format instruction based on model type
def get_formated_instruction(prompt, response, model_name, tokenizer):
    if "bloomz" in model_name:
        return_sample = prompt
    elif "gemma-2-9b-it" in model_name:
        messages = [{"role": "user", "content": prompt}]
        return_sample = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "llama3" in model_name.lower():
        return_sample = f'''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'''
    elif "command-r" in model_name:
        messages = [{"role": "user", "content": prompt}]
        return_sample = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif 'Qwen' in model_name:
        messages = [{"role": "user", "content": prompt}]
        return_sample = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return None, None
    
    # Tokenize only the prompt to get the index where the response starts
    prompt_model_inputs = tokenizer([return_sample], return_tensors="pt").to("cuda")
    index_response = len(prompt_model_inputs.input_ids[0])
    # Tokenize prompt+response to get the end index
    prompt_model_inputs_with_response = tokenizer([return_sample + str(response)], return_tensors="pt").to("cuda")
    index_response_end = len(prompt_model_inputs_with_response.input_ids[0])

    return prompt_model_inputs_with_response, index_response, index_response_end

# Main function to extract top neuron activations per input
def get_neuron():

    def get_model_response_file(
        filename=None,
        data_dir=None,
        model=None,
        country=None,
        language=None,
        prompt_no=None,
        template='{model}-{country}_{language}_{prompt_no}_result.csv'
        ):
        # Compose filename if not provided
        if filename is None:
            filename = template.replace('{model}', model).replace('{country}', country.replace(' ','_')).replace('{language}', language).replace('{prompt_no}', prompt_no)
            print(filename)
        if data_dir is None:
            assert 'ERROR: No data directory given' 
        # Read results CSV
        model_res_df = pd.read_csv(os.path.join(data_dir, model, filename), encoding='utf-8')
        return model_res_df

    print(f"===================Loading Model {args.model}=================")

    model_path = MODEL_PATHS[args.model]
    tokenizer, model = get_tokenizer_model(args.model, model_path, ".cache")
    model.generation_config.use_cache = False

    # Load result dataframe
    res_df = get_model_response_file(data_dir=args.output_dir, model=args.model, country=args.country, language=args.language, prompt_no=args.prompt_no)

    collate_fn = create_collate_fn(tokenizer)
    dataset = ResponseDataset(res_df, tokenizer, args.model)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Process batches
    for batch in tqdm(dataloader, desc='Processing batches'):
        batch_start_time = time.time()
        # Prepare activation dict for each layer
        act = {n: [] for n in range(model.config.num_hidden_layers)}
        
        # Register hooks to collect MLP activations
        def forward_hook(n):
            def fn(_, input, output):
                act[n].append(output.detach())
            return fn
        handle_act = [model.model.layers[n].mlp.act_fn.register_forward_hook(forward_hook(n)) for n in range(model.config.num_hidden_layers)]
        
        prompts = {
            'input_ids': batch['input_ids'].cuda(),
            'attention_mask': batch['attention_mask'].cuda()
        }
        prompt_indices = batch['prompt_indices']
        prompt_end_indices = batch['prompt_end_indices']
        IDs = batch['IDs']
        prompts_text = batch['prompts_text']
        responses_text = batch['responses_text']

        # Forward pass
        forward_start_time = time.time()
        with torch.no_grad():
            outputs = model(**prompts)
        # Remove hooks
        for handle in handle_act:
            handle.remove()
        forward_end_time = time.time()
        print(f"==========Time for forward pass: {forward_end_time - forward_start_time:.4f} seconds==========")
        
        # Collect activations for all layers
        activations = [act[n][0].half() for n in range(model.config.num_hidden_layers)]
        batch_size = activations[0].shape[0]
        seq_length = activations[0].shape[1]
        hidden_size = activations[0].shape[2]
        num_layers = model.config.num_hidden_layers

        scores_activate = [[[] for _ in range(batch_size)] for _ in range(num_layers)]
        
        # Mask for prompt-to-response positions
        position_mask = torch.zeros((batch_size, seq_length), dtype=torch.bool)
        for i in range(batch_size):
            prompt_index = prompt_indices[i].item()
            prompt_end_index = prompt_end_indices[i].item()
            position_mask[i, prompt_index:prompt_end_index] = True
        batch_indices, position_indices = position_mask.nonzero(as_tuple=True)

        score_start_time = time.time()
        # Collect activations at each position
        for layer in range(num_layers):
            activate_scores = activations[layer][batch_indices, position_indices, :]
            for idx, (b_idx, _) in enumerate(zip(batch_indices, position_indices)):
                scores_activate[layer][b_idx.item()].append(activate_scores[idx])
        score_end_time = time.time()
        print(f"==========Total time for score computation: {score_end_time - score_start_time:.4f} seconds==========")

        # For each sample in batch, select top neurons by activation
        top_activation_indices_per_case = [{} for _ in range(batch_size)]
        topk_start_time = time.time()
        for i in range(batch_size):
            for layer in range(num_layers):
                scores_activate_tensor = torch.stack(scores_activate[layer][i], dim=0)  # [num_positions, hidden_size]
                scores_activate_flat = scores_activate_tensor.flatten()
                total_neurons = scores_activate_flat.size(0)
                top_k = min(1000, total_neurons)
                # Select top-k neurons
                top_values_activation, top_indices_activation = torch.topk(scores_activate_flat, top_k)
                num_positions, hidden_size = scores_activate_tensor.shape
                top_positions_activation = [
                    (idx.item() // hidden_size, idx.item() % hidden_size) for idx in top_indices_activation
                ]
                top_activation_indices_per_case[i][layer] = {
                    'top_values': top_values_activation.cpu(),
                    'top_positions': top_positions_activation
                }
            # Prepare results for this sample
            results_activate = {
                'ID': IDs[i],
                'prompt': prompts_text[i],
                'response': responses_text[i],
                'top_neurons': []
            }
            for layer in range(num_layers):
                if layer in top_activation_indices_per_case[i]:
                    top_data = top_activation_indices_per_case[i][layer]
                    top_positions = top_data['top_positions']
                    top_values = top_data['top_values']
                    for idx, (position_idx, neuron_idx) in enumerate(top_positions):
                        score = top_values[idx].item()
                        results_activate['top_neurons'].append({
                            'layer': layer,
                            'position': position_idx,
                            'neuron': neuron_idx,
                            'score': score
                        })
            # Save results in JSONL
            output_neuron_dir = f'get_Neuron_result/{args.model}/'
            if not os.path.exists(output_neuron_dir):
                os.makedirs(output_neuron_dir)
            with open(f'{output_neuron_dir}/{args.model}-{args.country}_{args.language}_{args.prompt_no}_result_activate.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps(results_activate, ensure_ascii=False) + '\n')
        topk_end_time = time.time()
        print(f"==========Time for top-k selection: {topk_end_time - topk_start_time:.4f} seconds==========")

# Entry point for script, parse args and run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')
    parser.add_argument('--model', type=str, default="Llama3-8b-Instruct", help='Provide the model you want to use...')
    parser.add_argument('--country', type=str, default=None)
    parser.add_argument('--language', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./model_inference_results', help='Provide the directory for the output files to be saved.')
    parser.add_argument('--prompt_no', type=str, default="inst-8", help='Provide the prompt id (ex. inst-1, inst-2, pers-1, etc.')
    parser.add_argument('--last_only', type=bool, default=False, help="Will you only use the last token to find or using the fully ouput")
    args = parser.parse_args()

    get_neuron()