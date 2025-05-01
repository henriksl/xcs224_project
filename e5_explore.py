"""Module to explore e5 results"""

import json
import pandas as pd
import langcodes

def retrieve_single_model_results(path: str) -> dict:
    """Returns the dictionary with results from a single model"""

    tasknames: tuple[str, ...] = (
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
    )

    model_results = dict()

    for taskname in tasknames:
        full_path = f"{path}{taskname}.json"
        with open(full_path, encoding="utf-8") as json_file:
            model_results[taskname] = json.load(json_file)

    return model_results


def retrieve_results() -> dict:
    """Retrieves the json results from all models as dictionaries given MTEB output file structure"""

    path_start = "results/intfloat/multilingual-e5-"
    path_ends: tuple[str, ...] = ("small", "base", "large-instruct")
    model_codes: tuple[str, ...] = (
        "fd1525a9fd15316a2d503bf26ab031a61d056e98",
        "d13f1b27baf31030b7fd040960d60d909913633f",
        "baa7be480a7de1539afce709c8f13f833a510e0a",
    )

    results = dict()
    for path_end, model_code in zip(path_ends, model_codes):
        results[path_end] = retrieve_single_model_results(
            f"{path_start}{path_end}/intfloat__multilingual-e5-{path_end}/{model_code}/"
        )

    return results

def shorten_task_names(task_names: list[str]) -> list[str]:
    """Remove superflous 'Classification' in task names"""
    return [task_name.replace('Classification', '') for task_name in task_names]

def retrieve_avg_language_results(all_results: dict) -> pd.DataFrame:
    """
    Calculates a dataframe with avg test results over languages
    per task for each model
    """
    avg_dict = dict()
    for model in all_results.keys():
        for task in all_results[model].keys():
            test_results = all_results[model][task]['scores']['test']
            score = 0.0
            for test_result in test_results:
                score += test_result['f1']
            score /= len(test_results)
            try:
                avg_dict[task].append(score)
            except KeyError:
                avg_dict[task] = [score]

    # insert model sizes as index and transpose to task as y axis and model size x
    df = pd.DataFrame.from_dict(avg_dict)
    df.index = list(all_results.keys())
    df.columns = shorten_task_names(list(df.columns))
    return df.T


def save_per_lang_frame_results(per_lang_frame: pd.DataFrame) -> None:
    """Saves dataframe to csv"""
    per_lang_frame.to_csv('tex/avg_e5_results.csv', encoding='utf8',
                          float_format='%.2f')

def retrieve_save_avg(all_results: dict) -> None:
    """retrieve and save avg over language"""
    per_lang_frame =  retrieve_avg_language_results(all_results)
    save_per_lang_frame_results(per_lang_frame)


def remove_list_duplicates_keep_order(seq: list[str]) -> list[str]:
    """Removes duplicates in list and keeps first element for duplicates"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_languages_in_tasks(all_results: dict, tasks: list[str]) -> list[str]:
    """Get all unique languages in all_results within set tasks and return"""
    languages = list()
    for task_name in tasks:
        for model_name in all_results.keys():
            test_results = all_results[model_name][task_name]['scores']['test']
            for result in test_results:
                languages.extend(result['languages'])

    return remove_list_duplicates_keep_order(languages)


def parse_multilingual(all_results: dict, many_language_tasks: list[str]) -> dict:
    """
    Insert results into dictionary with keys task-model and values as f1 test scores
    sorted alphabetically according to language
    """
    result_dict = dict()

    for task_name in many_language_tasks:
        for model_name in all_results.keys():    
            task_model_name = f'{task_name}-{model_name}'
            result_dict[task_model_name]: list[tuple[float, str]] = list()

            # fetch list of tuples with f1 test score and language 
            test_results = all_results[model_name][task_name]['scores']['test']
            for result in test_results:
                if result['hf_subset'] == 'en-ext':
                    continue
                result_dict[task_model_name].append((result['f1'], result['languages'][0]))

            # sort list according to language alphabetically
            result_dict[task_model_name] = sorted(result_dict[f'{task_name}-{model_name}'], key=lambda x: x[1])

            # keep only scores in result dict
            result_dict[task_model_name] = [x[0] for x in result_dict[f'{task_name}-{model_name}']]
    
    return result_dict
        

def get_dataframe_model_many(all_results: dict, many_language_tasks: list[str]) -> pd.DataFrame:
    """
    Creates a dataframe with columns for task-model combinations and rows for languages.
    Used for massive datasets with 51 languages each.
    """

    task_languages = sorted(get_languages_in_tasks(all_results, many_language_tasks)) 
    result_dict: dict = parse_multilingual(all_results, many_language_tasks)
    
    df_many = pd.DataFrame.from_dict(result_dict)
    df_many.index = task_languages

    return df_many

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shorten and simplify for easier latex presentation"""
    # Convert to short language codes and only keep succesful conversions
    index = list(df.index)
    index = [langcodes.standardize_tag(elem) for elem in index]
    print(index)
    index = ['zh' if elem == 'cmo-Hans' else elem for elem in index]
    
    df.index = index
    print(df.index)
    df = df[df.index.str.len() < 3]

    df.columns = shorten_task_names(list(df.columns))
    print(df.index)
    if len(df.index) > 10:
        df = reduce_languages(df)
    
    # transpose
    return df.T

def reduce_languages(df: pd.DataFrame) -> pd.DataFrame:
    """Limit number of languages to display to fit in paper"""
    selected_languages = ['en', 'fr', 'de', 'esp', 'pl', 'zh', 'ru', 'nb', 'sv', 'sw', 'ur', 'my']
    df = df[df.index.isin(selected_languages)]
    return df
    
def retrieve_save_model_lang(all_results: dict) -> None:
    """
    Collects tasks into groups with identical languages,
    parses the results per task and model per language into a
    dataframe and saves the results to csv.
    """

    many_language_tasks = ['MassiveIntentClassification', 'MassiveScenarioClassification']
    mtop_tasks = ['MTOPDomainClassification', 'MTOPIntentClassification']
    amazon_counterfactual = ['AmazonCounterfactualClassification']
    amazon_reviews = ['AmazonReviewsClassification']
    
    df_many = get_dataframe_model_many(all_results, many_language_tasks)
    df_many = clean_df(df_many)
    df_many.to_csv('tex/massive_lang_e5.csv', encoding='utf8', float_format='%.2f')

    df_mtop = clean_df(get_dataframe_model_many(all_results, mtop_tasks))
    df_mtop.to_csv('tex/mtop_lang_e5.csv', encoding='utf8', float_format='%.2f')

    df_counter =  clean_df(get_dataframe_model_many(all_results, amazon_counterfactual))
    df_counter.to_csv('tex/counterfactural_lang_e5.csv', encoding='utf8', float_format='%.2f')

    df_reviews =  clean_df(get_dataframe_model_many(all_results, amazon_reviews))
    df_reviews.to_csv('tex/reviews_lang_e5.csv', encoding='utf8', float_format='%.2f')


def main() -> None:
    """
    Retrieves and saves results to csvs
    with results across all languages and 
    in a per language manner
    """
    
    all_results = retrieve_results()

    retrieve_save_avg(all_results)
    retrieve_save_model_lang(all_results)

if __name__ == '__main__':
    main()
