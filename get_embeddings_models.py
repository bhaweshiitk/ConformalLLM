import pickle
import os
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel

def get_qa_string(task_data, split, i):
    """Construct a QA string from the dataset."""
    q = task_data[split]['input'][i]
    a = task_data[split]['A'][i]
    b = task_data[split]['B'][i]
    c = task_data[split]['C'][i]
    d = task_data[split]['D'][i]
    return q + a + b + c + d

def get_embedding(text, model, tokenizer):
    """Generate embeddings for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

def save_to_pickle(data, filename):
    """Save data to a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    """Load data from a pickle file."""
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def main(task_list, embedding_model='bert-base-uncased'):
    # Dictionary of models and their corresponding tokenizers
    models = {
        'bert-base-uncased': (BertModel, BertTokenizer),
        'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer),
        'roberta-base': (RobertaModel, RobertaTokenizer)
    }

    # Load the selected model and tokenizer
    model_class, tokenizer_class = models[embedding_model]
    model = model_class.from_pretrained(embedding_model)
    tokenizer = tokenizer_class.from_pretrained(embedding_model)


    task_data_dict = {}
    embeddings_dict = {}

    for subject_name in task_list:
        task_data_dict[subject_name] = load_dataset('lukaemon/mmlu', subject_name)
        embeddings_dict[subject_name] = {}

    for subject_name, task_data in task_data_dict.items():
        print(f'Processing {subject_name}')
        for split in ['train', 'validation', 'test']:
            print(f'Processing {split} split')
            embeddings_dict[subject_name][split] = []
            for idx in range(len(task_data[split])):
                qa_string = get_qa_string(task_data, split, idx)
                embedding = get_embedding(qa_string, model, tokenizer)
                embeddings_dict[subject_name][split].append(embedding)

        save_to_pickle(embeddings_dict, f'embeddings_dict_{embedding_model}.pkl')


if __name__ == "__main__":
    model_choice = 'bert-base-uncased'  # Change this to 'distilbert-base-uncased' or 'roberta-base' as needed
    task_list = [
        'college_computer_science', 'formal_logic', 'high_school_computer_science',
        'computer_security', 'machine_learning',

        'clinical_knowledge', 'high_school_biology', 'anatomy', 'college_chemistry',
        'college_medicine', 'professional_medicine',

        'business_ethics', 'professional_accounting', 'public_relations',
        'management', 'marketing'
    ]
    filename = 'embeddings_dict_' + model_choice + '.pkl'

    if not os.path.exists(filename):
        main(task_list, embedding_model=model_choice)

    embeddings_dict = load_from_pickle(filename)

    """
    embeddings_dict: A dictionary containing embeddings for various subjects and their respective data splits.

    Structure:
    {
        'subject_name_1': {
            'train': [embedding_1, embedding_2, ...],
            'validation': [embedding_1, embedding_2, ...],
            'test': [embedding_1, embedding_2, ...]
        },
        'subject_name_2': {
            'train': [embedding_1, embedding_2, ...],
            'validation': [embedding_1, embedding_2, ...],
            'test': [embedding_1, embedding_2, ...]
        },
        ...
    }

    - Each key at the top level corresponds to a subject name from the task_list.
    - For each subject, there's a nested dictionary that contains embeddings for different data splits:
     'train', 'validation', and 'test'.
    - For each data split, there's a list of embeddings. Each embedding in the list corresponds to a QA
      string from the dataset.
    - Each embedding is a vector (numpy array) representing the [CLS] token's output from the BERT model
      for the respective QA string.
    """
    for subject_name in task_list:
        for split in ['train', 'validation', 'test']:
            print(f'subject name: {subject_name}, split: {split}, \
             length {len(embeddings_dict[subject_name][split])}, \
             shape: {embeddings_dict[subject_name][split][0].shape}')
