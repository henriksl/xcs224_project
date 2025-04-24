"""Script used to evaluate e5 on set classifcation task for mini, base and large-instruct models"""

import mteb
from mteb.overview import MTEBTasks

def evaluate_e5() -> None:
    """Function used to evaluate and save all results from e5 mini, base and large-instruct run"""

    model_names: list[str] = ['intfloat/multilingual-e5-small', 'intfloat/multilingual-e5-base', 'intfloat/multilingual-e5-base']
    
    task_names = ['MassiveIntentClassification', 'MassiveScenarioClassification',
                  'AmazonCounterfactualClassification', 'AmazonReviewsClassification',
                  'MTOPDomainClassification', 'MTOPIntentClassification']

    tasks: MTEBTasks = mteb.get_tasks(tasks=task_names)

    def evaluate_model(model_name: str) -> None:
        """Evaluates single model on above tasks"""
        model: mteb.Encoder = mteb.get_model(model_name)
        evaluation = mteb.MTEB(tasks=tasks)
        _ = evaluation.run(model, output_folder=f"results/{model_name}", verbosity=2)

    for model_name in model_names:
        evaluate_model(model_name)

if __name__ == '__main__':
    evaluate_e5()