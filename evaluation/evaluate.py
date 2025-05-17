from evaluation_utils import *
from exact_match import *

def evaluate_all_metrics(
    model,country,language,
    prompt_no,response_dir,annotation_dir,mc_dir,
    id_col,q_col,r_col,annotations_key,
    eval_res_filename,annotation_template='{country}_data.json',
    masked=None
    ):
    
    if not os.path.exists(eval_res_filename):
        write_csv_row(['model','country','language','prompt_no','eval_method','score'],eval_res_filename)
    
    res_df = get_model_response_file(data_dir=response_dir,model=model,country=country,language=language,prompt_no=prompt_no,masked=masked)
      
    real_annotation = get_annotations(data_dir=annotation_dir,country=country,template=annotation_template)
    
    sem_b,sem_w,res_df = soft_exact_match(country=country,language=COUNTRY_LANG[language],annotation_dict=real_annotation,response_df=res_df,id_col=id_col,r_col=r_col,annotations_key=annotations_key)
    
    if masked == None:
        write_csv_row([model,country,language,prompt_no,'SEM-B',sem_b],eval_res_filename)
        write_csv_row([model,country,language,prompt_no,'SEM-W',sem_w],eval_res_filename)
    else:
        write_csv_row([f"{model}_{masked}",country,language,prompt_no,'SEM-B',sem_b],eval_res_filename)
        write_csv_row([f"{model}_{masked}",country,language,prompt_no,'SEM-W',sem_w],eval_res_filename)
    
    if masked == None:
        eval_result_file_path = os.path.join(response_dir, f'{model}/{model}_{country}_{language}_{prompt_no}_response_score.csv')
    else:
        eval_result_file_path = os.path.join(response_dir, f'{model}/{model}_{country}_{language}_{prompt_no}_{masked}_response_score.csv')

    # Check if the file exists, and if so, delete it
    if os.path.exists(eval_result_file_path):
        os.remove(eval_result_file_path)
    res_df.to_csv(eval_result_file_path, index=False, encoding='utf-8')
    

    # leave the latest result if duplicated
    # Read the file as pd.DataFrame
    df = pd.read_csv(eval_res_filename)

    # Delete duplicate lines regarding model, country, language, prompt_no, eval_method
    df.drop_duplicates(subset=['model', 'country', 'language', 'prompt_no', 'eval_method'], keep='last', inplace=True)

    # Write the modified DataFrame back to the file
    df.to_csv(eval_res_filename, index=False, encoding='utf-8')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')
    parser.add_argument('--model',type=str,
                        help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable. If you want to test on multiple models, provide multiple model names with ", " between each (e.g., "gpt-4-0125-preview, aya-101").')
    parser.add_argument('--language',type=str,default=None,
                        help='Provide the language you want to test on. Check and choose from the first values of the LANG_COUNTRY variable. If you want to test on multiple languages, provide multiple languages with ", " between each (e.g., "English, Korean").')
    parser.add_argument('--country',type=str,default=None,
                        help='Provide the country you want to test on. Check and choose from the second values of the LANG_COUNTRY variable. If you want to test on multiple countries, provide multiple countries with ", " between each (e.g., "UK, South Korea"). Make sure you have the same number of countries and languages provided. The language-country pair do not have to be identical with the pairs within the LANG_COUNTRY variable.')
    parser.add_argument('--prompt_no',type=str,default=None,
                        help='Provide the propmt id (ex. inst-1, inst-2, pers-1, etc.')
    
    parser.add_argument('--id_col',type=str,default=None,
                        help='Provide the column name from the LLM response csv file name with question IDs.') 
    parser.add_argument('--question_col',type=str,default=None,
                        help='Provide the column name from the LLM response csv file name with questions.')
    parser.add_argument('--response_col',type=str,default=None,
                        help='Provide the column name from the LLM response csv file name with LLM responses.') 

    parser.add_argument('--response_dir',type=str,default='../model_inference_results',
                        help='Provide the directory for the output files to be saved.')
    parser.add_argument('--annotation_dir',type=str,default='../data/annotations/',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--mc_dir',type=str,default='./mc_data',
                        help='Provide the directory for the multiple choice result files.')
    parser.add_argument('--annotation_filename',type=str,default='{country}_data.json',)
    parser.add_argument('--annotations_key',type=str,default='annotations',
                        help='Provide the key for the annotations in the annotation file.')
    parser.add_argument('--evaluation_result_file',type=str,default='evaluation_results.csv',
                        help='Provide the filename for the evaluation result file.')
    parser.add_argument('--masked',type=str,default=None)
    parser.add_argument('--layer', type=int, nargs='+', default=None ,help='List of layers to evaluate')

    
    args = parser.parse_args()
    if args.language == args.country:
        args.question_col = "Translation"
    elif args.language == "UK" or args.language == "US":
        args.question_col = "Question"
    else:
        args.question_col = args.language

    if args.language == args.country or args.language == "UK" or args.language =="US":
        args.annotation_filename = f"{args.country}_data.json"
    else:
        args.annotation_dir = "../data/annotations_multi/"
        args.annotation_filename = f"{args.country}_{args.language}_data.json"

   
    if args.masked == None:
        evaluate_all_metrics(model=args.model,country=args.country,language=args.language,prompt_no=args.prompt_no,response_dir=args.response_dir,annotation_dir=args.annotation_dir,mc_dir=args.mc_dir,id_col=args.id_col,q_col=args.question_col,r_col=args.response_col,eval_res_filename=args.evaluation_result_file,annotations_key=args.annotations_key,annotation_template=args.annotation_filename) 
    else:
        if args.layer is not None:
            layer_str = ""
            for layer in args.layer: layer_str += str(layer)
            evaluate_all_metrics(model=args.model,country=args.country,language=args.language,prompt_no=args.prompt_no,response_dir=args.response_dir,annotation_dir=args.annotation_dir,mc_dir=args.mc_dir,id_col=args.id_col,q_col=args.question_col,r_col=args.response_col,eval_res_filename=args.evaluation_result_file,annotations_key=args.annotations_key,annotation_template=args.annotation_filename, masked=f"{args.masked}_{layer_str}") 
            evaluate_all_metrics(model=args.model,country=args.country,language=args.language,prompt_no=args.prompt_no,response_dir=args.response_dir,annotation_dir=args.annotation_dir,mc_dir=args.mc_dir,id_col=args.id_col,q_col=args.question_col,r_col=args.response_col,eval_res_filename=args.evaluation_result_file,annotations_key=args.annotations_key,annotation_template=args.annotation_filename, masked=f"{args.masked}_{layer_str}_random") 
            
        else:
            evaluate_all_metrics(model=args.model,country=args.country,language=args.language,prompt_no=args.prompt_no,response_dir=args.response_dir,annotation_dir=args.annotation_dir,mc_dir=args.mc_dir,id_col=args.id_col,q_col=args.question_col,r_col=args.response_col,eval_res_filename=args.evaluation_result_file,annotations_key=args.annotations_key,annotation_template=args.annotation_filename, masked=args.masked) 
            evaluate_all_metrics(model=args.model,country=args.country,language=args.language,prompt_no=args.prompt_no,response_dir=args.response_dir,annotation_dir=args.annotation_dir,mc_dir=args.mc_dir,id_col=args.id_col,q_col=args.question_col,r_col=args.response_col,eval_res_filename=args.evaluation_result_file,annotations_key=args.annotations_key,annotation_template=args.annotation_filename, masked=f"{args.masked}_random") 