from utils import *
from utils_neuron import *

parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')
parser.add_argument('--model',type=str,
                    help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable. If you want to test on multiple models, provide multiple model names with ", " between each (e.g., "gpt-4-0125-preview, aya-101").')
parser.add_argument('--language',type=str,default=None,
                    help='Provide the language you want to test on. Check and choose from the first values of the LANG_COUNTRY variable. If you want to test on multiple languages, provide multiple languages with ", " between each (e.g., "English, Korean").')
parser.add_argument('--country',type=str,default=None,
                    help='Provide the country you want to test on. Check and choose from the second values of the LANG_COUNTRY variable. If you want to test on multiple countries, provide multiple countries with ", " between each (e.g., "UK, South Korea"). Make sure you have the same number of countries and languages provided. The language-country pair do not have to be identical with the pairs within the LANG_COUNTRY variable.')
parser.add_argument('--question_dir',type=str,default=None,
                    help='Provide the directory name with (translated) questions.')
parser.add_argument('--question_file',type=str,default=None,
                    help='Provide the csv file name with (translated) questions.')
parser.add_argument('--question_col',type=str,default=None,
                    help='Provide the column name from the given csv file name with (translated) questions.')
parser.add_argument('--prompt_dir',type=str,default=None,
                    help='Provide the directory where the propmts are saved.')
parser.add_argument('--prompt_file',type=str,default=None,
                    help='Provide the name of the csv file where the propmts are saved.')
parser.add_argument('--prompt_no',type=str,default=None,
                    help='Provide the propmt id (ex. inst-1, inst-2, pers-1, etc.)')
parser.add_argument('--id_col',type=str,default="ID",
                    help='Provide the column name from the given csv file name with question IDs.')
parser.add_argument('--output_dir',type=str,default='./model_inference_results',
                    help='Provide the directory for the output files to be saved.')
parser.add_argument('--output_file',type=str,default=None,
                    help='Provide the name of the output file.')
parser.add_argument('--model_cache_dir',type=str,default='.cache',
                    help='Provide the directory saving model caches.')
parser.add_argument("--gpt_azure", default=True)
parser.add_argument('--temperature',type=float,default=0.0,
                    help='Provide generation temperature for GPT models.')
parser.add_argument('--top_p',type=int,default=0,
                    help='Provide generation top_p for GPT models.')
parser.add_argument('--masked',type=str,default="If masking the key neurons to have abliation study please provide the neuron file path")
parser.add_argument('--layer', type=int, nargs='+', default=None ,help='List of layers to evaluate')

args = parser.parse_args()

def make_prompt(question,prompt_no,language,country,prompt_sheet):
    prompt = prompt_sheet[prompt_sheet['id']==prompt_no]
    if language == 'English':
        prompt = prompt['English'].values[0]
    else:
        prompt = prompt['Translation'].values[0]

    return prompt.replace('{q}',question)

def generate_response(model_name,model_path,tokenizer,model,language,country,q_df,q_col,id_col,output_dir,prompt_no=None, masked=None):
    replace_country_flag = False
    if q_col == "Question":
        replace_country_flag = True
        
    if q_col != "Question" and q_col != "Translation":
        prompt_sheet = pd.read_csv(os.path.join(args.prompt_dir,f'{q_col}_prompts.csv'),encoding='utf-8')
        if masked == None:
            output_filename = os.path.join(output_dir,model_name, f"{model_name}-{country}_{language}_{prompt_no}_result.csv")
        else:
            output_filename = os.path.join(output_dir,model_name, f"{model_name}-{country}_{language}_{prompt_no}_{masked}_result.csv")
    else:
        prompt_sheet = pd.read_csv(os.path.join(args.prompt_dir,f'{language}_prompts.csv'),encoding='utf-8')
        if q_col == "Translation":
            if masked == None:
                output_filename = os.path.join(output_dir, model_name, f"{model_name}-{country}_{language}_{prompt_no}_result.csv")
            else:
                output_filename = os.path.join(output_dir, model_name, f"{model_name}-{country}_{language}_{prompt_no}_{masked}_result.csv")
        else:
            if masked == None:
                output_filename = os.path.join(output_dir, model_name, f"{model_name}-{country}_{language}_{prompt_no}_result.csv")
            else:
                output_filename = os.path.join(output_dir, model_name, f"{model_name}-{country}_{language}_{prompt_no}_{masked}_result.csv")
            

    print(q_df[[id_col,q_col]])
    
    
    guid_list = set()
    if os.path.exists(output_filename):
        already = pd.read_csv(output_filename)
        guid_list = set(already[id_col])
        print(already)
    else:        
        os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
        write_csv_row([id_col,q_col,'prompt','response','prompt_no'],output_filename)
      
    pb = tqdm(q_df.iterrows(),desc=model_name,total=len(q_df))
    for _,d in pb:
        q = d[q_col]
        guid = d[id_col]
        pb.set_postfix({'ID':guid})
        
        if guid in guid_list:
            continue
       
        if replace_country_flag:
            q = replace_country_name(q,country.replace('_',' '))
       
        if prompt_no is not None:
            if q_col == "Question":
                prompt = make_prompt(q, prompt_no, "English", country, prompt_sheet)
            else:
                prompt = make_prompt(q, prompt_no, language, country, prompt_sheet)
        else:
            prompt = q
            
        
        response = get_model_response(model_name,prompt,model,tokenizer,temperature=args.temperature,top_p=args.top_p,gpt_azure=args.gpt_azure)
            
        write_csv_row([guid,q,prompt,response,prompt_no],output_filename)
        
    del guid_list
            
def get_response_from_all():
    models = args.model
    languages = args.language
    countries = args.country
    question_dir = args.question_dir
    question_file = args.question_file
    if languages == countries:
        question_col = "Translation"
    elif languages == "UK" or languages == "US":
        question_col = "Question"
    else:
        question_col = languages
    prompt_no = args.prompt_no
    id_col = args.id_col
    output_dir = args.output_dir
    azure = args.gpt_azure
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
        
    def get_questions(language,country):
        questions_df = pd.read_csv(os.path.join(question_dir,f'{country}_questions.csv'),encoding='utf-8')

        return questions_df
    
    
    def generate_response_per_model(model_name):
        model_path = MODEL_PATHS[model_name]
        print("========",model_name, model_path, "============")


        tokenizer,model = get_tokenizer_model(model_name,model_path,args.model_cache_dir)
        questions = get_questions(languages,countries)

        if args.masked != None:
            with open(f"neuron/{models}/China_UK_{args.masked}.json", 'r') as f:
                neuron_list = json.load(f)

            if args.layer is not None:
               
                layers_set = set(args.layer)
                filtered_neuron_list = [neuron for neuron in neuron_list if neuron[0] in layers_set]
                neuron_set = set(tuple(neuron) for neuron in filtered_neuron_list)
                print(len(neuron_set))
                model_zero_specified, model_zero_random = zero_neurons_and_random(model, neuron_set)

                layer_str = ""
                for layer in args.layer: layer_str += str(layer)
                generate_response(model_name,model_path,tokenizer,model_zero_specified,languages,countries,questions,question_col,id_col,output_dir,prompt_no=prompt_no, masked=f"{args.masked}_{layer_str}")
                generate_response(model_name,model_path,tokenizer,model_zero_random,languages,countries,questions,question_col,id_col,output_dir,prompt_no=prompt_no, masked=f"{args.masked}_{layer_str}_random")


            else:
                neuron_set = set(tuple(neuron) for neuron in neuron_list)

                model_zero_specified, model_zero_random = zero_neurons_and_random(model, neuron_set)
                generate_response(model_name,model_path,tokenizer,model_zero_specified,languages,countries,questions,question_col,id_col,output_dir,prompt_no=prompt_no, masked=args.masked)
                generate_response(model_name,model_path,tokenizer,model_zero_random,languages,countries,questions,question_col,id_col,output_dir,prompt_no=prompt_no, masked=f"{args.masked}_random")
    
        else:
            generate_response(model_name,model_path,tokenizer,model,languages,countries,questions,question_col,id_col,output_dir,prompt_no=prompt_no)

            
    
    generate_response_per_model(models)

 
if __name__ == "__main__":
    get_response_from_all()    