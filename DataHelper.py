import json  # Used for reading and parsing JSON files.
import transformers  # (Not used explicitly in the function, but could be useful for tokenization later.)

def get_dataset(dataset_name, sep_token):
    '''
    This function retrieves a dataset based on the `dataset_name` and processes it into a format suitable for training 
    and testing models. It handles multiple datasets like SemEval, ACL-ARC, AGNews, and few-shot variants.

    Args:
        dataset_name (str): The name of the dataset to load. Can be 'laptop_sup', 'restaurant_sup', 'acl_sup', 'agnews_sup', etc.
        sep_token (str): A separator token (e.g., '<sep>') used to join different parts of the text.

    Returns:
        dataset (datasets.DatasetDict or datasets.Dataset): A Hugging Face `datasets.DatasetDict` containing the train 
        and test datasets, or a merged dataset in case of few-shot datasets.
    '''
    
    import datasets  # Importing the Hugging Face `datasets` library to load and process datasets.
    
    dataset = {}  # Initialize an empty dictionary to store the dataset.

    # Check if the dataset_name is a string
    if isinstance(dataset_name, str):
        
        # Handle 'laptop_sup' and 'restaurant_sup' datasets (SemEval14)
        if dataset_name == 'laptop_sup' or dataset_name == 'restaurant_sup':
            dataset_name = dataset_name.replace('_sup', '')  # Remove '_sup' from the name to match folder names.

            # Path to the training dataset (assumed to be in JSON format).
            train_path = f'./SemEval14-{dataset_name}/train.json'
            with open(train_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)  # Load the training data from JSON.

            train_text = []  # List to store training text.
            train_label = []  # List to store training labels.
            
            # Process each entry in the training data.
            for item in train_data.values():
                train_text.append(item['term'] + item['sentence'])  # Concatenate 'term' and 'sentence' for the text.
                
                # Convert polarity labels to integers: positive -> 1, negative -> 0, neutral -> 2.
                if item['polarity'] == 'positive':
                    i = 1
                elif item['polarity'] == 'negative':
                    i = 0
                else:
                    i = 2
                train_label.append(i)  # Append the label to the list.

            # Create a dictionary for the training data.
            train_data = {"text": train_text, "label": train_label}
            # Convert this dictionary to a Hugging Face `datasets.Dataset`.
            train_dataset = datasets.Dataset.from_dict(train_data)

            # Path to the test dataset (assumed to be in JSON format).
            test_path = f'./SemEval14-{dataset_name}/test.json'
            with open(test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)  # Load the test data from JSON.

            test_text = []  # List to store test text.
            test_label = []  # List to store test labels.
            
            # Process each entry in the test data.
            for item in test_data.values():
                # Concatenate 'term', `sep_token`, and 'sentence'.
                test_text.append(item['term'] + sep_token + item['sentence'])
                
                # Convert polarity labels to integers.
                if item['polarity'] == 'positive':
                    i = 1
                elif item['polarity'] == 'negative':
                    i = 0
                else:
                    i = 2
                test_label.append(i)

            # Create a dictionary for the test data.
            test_data = {"text": test_text, "label": test_label}
            # Convert this dictionary to a Hugging Face `datasets.Dataset`.
            test_dataset = datasets.Dataset.from_dict(test_data)

            # Create a DatasetDict with both train and test datasets.
            dataset = datasets.DatasetDict({'train': train_dataset, 'test': test_dataset})

            print(dataset['test'][0])  # Print the first test example for debugging.
        
        # Handle 'acl_sup' dataset (ACL-ARC)
        if dataset_name == 'acl_sup':
            train_path = './ACL-ARC/train.jsonl'  # Path to the training dataset (assumed to be in JSONL format).
            train_text = []  # List for training text.
            train_label = []  # List for training labels.
            label2id = {}
            cur_id = 0 

            with open(train_path, 'r') as f:
                lines = f.readlines()  # Read all lines from the JSONL file.
                
                # Process each line of the training file.
                for line in lines:
                    line = json.loads(line)  # Parse each line as JSON.
                    train_text.append(line['text'])  # Append the sentence.
                    if line["label"] not in label2id:
                        label2id[line["label"]] = cur_id
                        cur_id += 1
                    train_label.append(label2id[line["label"]])  # Append whether it has a citation (label).

            print("current label2id:", cur_id)
            test_path = './ACL-ARC/test.jsonl'  # Path to the test dataset.
            test_text = []  # List for test text.
            test_label = []  # List for test labels.

            with open(test_path, 'r') as f:
                for line in f:
                    # Similar processing for test data.
                    test_text.append(json.loads(line)['text'])
                    test_label.append(label2id[json.loads(line)["label"]])

            # Create dictionaries for train and test data.
            train_data = {"text": train_text, "label": train_label}
            test_data = {"text": test_text, "label": test_label}

            # Convert these dictionaries to Hugging Face `datasets.Dataset`.
            train_dataset = datasets.Dataset.from_dict(train_data)
            test_dataset = datasets.Dataset.from_dict(test_data)

            # Create a DatasetDict for train and test.
            dataset = datasets.DatasetDict({'train': train_dataset, 'test': test_dataset})
            print(dataset['test'][0])  # Print the first test example for debugging.

        # Handle 'agnews_sup' dataset (AGNews CSV)
        if dataset_name == "agnews_sup":
            dataset_path = 'AGNews/test.csv'  # Path to the dataset (assumed to be a CSV file).
            texts = []  # List to store the combined text.
            labels = []  # List to store labels.

            # Read the CSV file.
            with open(dataset_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()  # Read all lines.
                
                # Process each line in the CSV.
                for line in lines:
                    line = line.split(',')  # Split CSV line by commas.
                    labels.append(int(line[0]))  # First column is the label.
                    texts.append(line[1] + f"{sep_token}" + line[2])  # Combine the article title and description.

            # Create a dictionary for the dataset.
            dataset_data = {"text": texts, "label": labels}
            # Convert this dictionary to a Hugging Face `datasets.Dataset`.
            dataset = datasets.Dataset.from_dict(dataset_data)

            # Split the dataset into train and test (90% train, 10% test) with a fixed seed.
            dataset = dataset.train_test_split(test_size=0.1, seed=2022)
            print(dataset['test'][0])  # Print the first test example for debugging.

        # Handle few-shot datasets (suffix 'fs')
        if dataset_name.endswith('fs'):
            dataset_name = dataset_name.replace('fs', 'sup')  # Replace 'fs' with 'sup' to use the supervised dataset.
            dataset = get_dataset(dataset_name, sep_token)  # Recursively call `get_dataset` for the supervised version.
            dataset['train'] = dataset['train'][:32]
            dataset['test'] = dataset['test'][:32] # Keep only the first 32 examples (few-shot scenario).
            print(dataset['test'][0])  # Print the first test example for debugging.

    # implement the aggregation of multiple datasets
    if isinstance(dataset_name, list):
        dataset = []
        biggest_label = 0
        for name in dataset_name:
            # Recursively call get_dataset for each dataset in the list.
            new_dataset = get_dataset(name, sep_token)
            def modify_labels(example):
            # 假设你想将标签 0 改为 1，标签 1 改为 0
                example['label'] += biggest_label
                if example['label'] > biggest_label:
                    biggest_label = example['label']
                biggest_label += 1
                return example
            # 使用 map 方法应用修改标签的函数
            new_dataset = new_dataset.map(modify_labels)
            dataset.append(new_dataset)
        
        # Concatenate the datasets into one.
        dataset = datasets.concatenate_datasets(dataset)

    return dataset  # Return the final dataset.


if __name__ == "__main__":
    # Example usage: Load the AGNews dataset with a comma as the separator.
    dataset = get_dataset("laptop_sup", ',')
    
    # Print the first 10 test texts for verification.
    print(dataset['test']['text'][:10])
    
    # Print the entire dataset structure for debugging.
    print(dataset)