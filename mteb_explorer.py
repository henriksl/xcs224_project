import mteb
import mteb.abstasks
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.errors import GatedRepoError

xlm_r_iso_639_2 = [
    'afr', 'amh', 'ara', 'asm', 'aze', 'bel', 'bul', 'ben', 'bod', 'bre', 'bos', 'cat',
    'ceb', 'ces', 'cym', 'dan', 'deu', 'ell', 'eng', 'spa', 'est', 'eus', 'fas', 'fin',
    'fra', 'fry', 'gle', 'gla', 'glg', 'guj', 'hau', 'heb', 'hin', 'hrv', 'hat', 'hun',
    'hye', 'ind', 'ibo', 'isl', 'ita', 'jpn', 'jav', 'kat', 'kaz', 'khm', 'kan', 'kor',
    'kur', 'kir', 'lat', 'ltz', 'lao', 'lit', 'lav', 'mlg', 'mkd', 'mal', 'mon', 'mar',
    'msa', 'mlt', 'mya', 'nep', 'nld', 'nor', 'orm', 'ori', 'pan', 'pol', 'pus', 'por',
    'que', 'ron', 'rus', 'kin', 'sin', 'slk', 'slv', 'smo', 'sna', 'som', 'sqi', 'srp',
    'sun', 'swe', 'swa', 'tam', 'tel', 'tha', 'tgl', 'tur', 'uig', 'ukr', 'urd', 'uzb',
    'vie', 'wol', 'xho', 'yid', 'zho', 'zul'
]

def get_xlm_language_classification_tasks() -> dict[str, list[mteb.abstasks.AbsTask]]:
    """
    Fetch classification tasks where all the languages in the task are included
    in XLM-R/mDebertaV3 pretraining. 
    """

    return {language: mteb.get_tasks(tasks=['MassiveIntentClassification', 'MassiveScenarioClassification', 'AmazonCounterfactualClassification', 'AmazonReviewsClassification', 'MTOPDomainClassification', 'MTOPIntentClassification'], 
                                        languages=[language], exclusive_language_filter=True) for language in xlm_r_iso_639_2}
   
def get_task_languages(tasks: list[mteb.abstasks.AbsTask]) -> set[str]:
    """
    Return a set of all languages in input tasks list
    """
    languages = set()
    for task in tasks:
        languages.update(task.languages)
    return languages

def print_examples(task: mteb.abstasks.AbsTask) -> None:
    task.load_data()
    train_data = task.dataset['train']
    print(dir(train_data))
   

def main():

    tasks = mteb.get_tasks(tasks=['MassiveIntentClassification', 'MassiveScenarioClassification', 
                                  'AmazonCounterfactualClassification', 'AmazonReviewsClassification',
                                  'MTOPDomainClassification', 'MTOPIntentClassification'])
    for task in tasks:
        print(task, task.languages)
        task.load_data()
        
        print(task.dataset.keys())

    return

if __name__ == '__main__':
    main()