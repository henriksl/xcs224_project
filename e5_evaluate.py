"""Script used to evaluate e5 on set classifcation task for mini, base and large-instruct models"""

import argparse

import mteb
from mteb.overview import MTEBTasks

def evaluate_e5(model_index: int) -> None:
    """Function used to evaluate and save all results from e5 mini, base and large-instruct run"""

    model_names: list[str] = ['intfloat/multilingual-e5-small', 'intfloat/multilingual-e5-base', 'intfloat/multilingual-e5-large-instruct']
    
    task_names = ['MassiveIntentClassification', 'MassiveScenarioClassification',
                  'AmazonCounterfactualClassification', 'AmazonReviewsClassification',
                  'MTOPDomainClassification', 'MTOPIntentClassification']

    tasks: MTEBTasks = mteb.get_tasks(tasks=task_names)

    def evaluate_model(model_name: str) -> None:
        """Evaluates single model on above tasks"""
        model: mteb.Encoder = mteb.get_model(model_name)
        evaluation = mteb.MTEB(tasks=tasks)
        _ = evaluation.run(model, output_folder=f"results/{model_name}", verbosity=2)

    selected_model = model_names[model_index]
    evaluate_model(selected_model)


def get_model_index() -> int:
    """
    Use model index to run script with different models
    0 - mini
    1 - base
    2 - large-instruct
    E.g. run
        python e5_evaluate.py 0
    to use mini model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    return int(args.model)


if __name__ == '__main__':
    _model_index = get_model_index()
    evaluate_e5(_model_index)
