"""Script to perform zero shot classification with nli models"""

import argparse
import json

import evaluate
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset, get_dataset_config_names

def remove_undesirable_configs(configs: list[str]) -> list[str]:

    undesirable_configs = ['all_languages', 'default']
    for undesirable in undesirable_configs:
        try:
            configs.remove(undesirable)
        except ValueError:
            pass
    
    return configs

def evaluate_f1_text(labels: list[str], predictions: list[str], references: list[str]) -> float:
    """Convert to indices and then calucalte f1"""

    label_index_dict = dict([(label, index) for index, label in enumerate(sorted(set(labels)))])

    predictions_indices = [label_index_dict[label] for label in predictions]
    references_indices = [label_index_dict[label] for label in references]

    f1_metric = evaluate.load("f1")
    results = f1_metric.compute(predictions=predictions_indices, references=references_indices, average='macro')
    return results["f1"]


def perform_evaluations(model_name: str, batch_size: int) -> dict[str, dict[str, float]]:
    """
    Evaluates model per task and config/language and returns macro f1 scores 
    per task per language.    
    """

    classifier = pipeline("zero-shot-classification", model=model_name)

    dataset_names = [
        "mteb/amazon_massive_intent",
        "mteb/amazon_massive_scenario",
        "mteb/amazon_counterfactual",
        "mteb/amazon_reviews_multi",
        "mteb/mtop_domain",
        "mteb/mtop_intent",
    ]

    results = dict()

    for dataset_name in dataset_names:

        results[dataset_name] = dict()

        configs = get_dataset_config_names(dataset_name)
        configs = remove_undesirable_configs(configs)
        labels: None | list[str] = None
        
        for index, config in enumerate(configs):

            dataset = load_dataset(dataset_name, config, split='test')

            if index == 0:
                labels = list(set(dataset['label']))

            predictions = list()
            for output in tqdm(classifier(KeyDataset(dataset, 'text'), labels, multilabel=False, batch_size=batch_size)):
                predictions.append(output['labels'][0])
                
            references = dataset['label']

            f1 = evaluate_f1_text(labels, predictions, references)
            results[dataset_name][config] = f1
            print(f'{dataset_name}_{config} f1: {f1}')

    return results

def save_results(model_name: str, results: dict) -> None:
    """Save results to json"""
    model_name = model_name.replace('/','_')
    model_name = model_name.replace('-','_')
    filename = f'results/zero/{model_name}.json'

    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(results, fp)

def evaluate_zero_nli(model_index: int) -> None:
    """Used to evaulate model index across selected MTEB tasks and saves to json"""

    model_name = [
        "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "joeddav/xlm-roberta-large-xnli",
    ][model_index]

    batch_size = [256, 128, 128][model_index]

    results: dict[str, dict[str, float]] = perform_evaluations(model_name, batch_size)

    save_results(model_name, results)


def get_model_index() -> int:
    """
    Use model index to run script with different models
    0 - mini
    1 - base
    2 - large
    E.g. run
        python nli_zero_evaluate.py 0
    to use mini model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    return int(args.model)

if __name__ == '__main__':
    _model_index = get_model_index()
    evaluate_zero_nli(_model_index)
