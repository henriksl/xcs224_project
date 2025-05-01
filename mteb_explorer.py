"""Script to explore datasets and languages in MTEB benchmark"""

import mteb
import mteb.abstasks

def main():

    tasks: list[mteb.abstasks.AbsTask] = mteb.get_tasks(tasks=['MassiveIntentClassification', 'MassiveScenarioClassification', 
                                  'AmazonCounterfactualClassification', 'AmazonReviewsClassification',
                                  'MTOPDomainClassification', 'MTOPIntentClassification'])
    
    for task in tasks:
        print(task, task.languages)
        task.load_data()
        print(task.dataset.keys())

    return

if __name__ == '__main__':
    main()