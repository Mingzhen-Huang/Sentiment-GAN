# inbuilt lib imports:
from collections import Counter
from typing import List, Dict, Tuple, Any
import json
import os
import zipfile

# external lib imports:
import numpy as np
from tqdm import tqdm
import spacy


nlp = spacy.load("en_core_web_sm", disable = ['ner', 'tagger', 'parser', 'textcat'])

def read_instances(data_file_path: str,
                   max_allowed_num_tokens: int = 150) -> List[Dict]:
    """
    Reads raw classification dataset from a file and returns a list
    of dicts where each dict defines an instance.
    Parameters
    ----------
    data_file_path : ``str``
        Path to data to be read.
    max_allowed_num_tokens : ``int``
        Maximum number of tokens allowed in the classification instance.
    """
    instances = []
    clause_max_len = 0
    with open(data_file_path) as file:
        for line in tqdm(file.readlines()):
            instance = {"text": None}
            instance["text"] = line.rstrip("\n")[:-1].replace(',', '').strip()
            text = instance["text"]
            tokens = [token.text.lower() for token in nlp.tokenizer(text)][:max_allowed_num_tokens]
            instance["text_tokens"] = tokens
            instance.pop("text")
            if len(instance["text_tokens"]) != 1:
                instances.append(instance)
            if len(instance["text_tokens"]) > clause_max_len:
                clause_max_len = len(instance["text_tokens"])
    for instance in instances:
        text_tokens_length = len(instance["text_tokens"])
        if text_tokens_length < clause_max_len:
            instance["text_tokens"].extend(["@PAD@"]*(clause_max_len-text_tokens_length))
    # print(instances)
    # print(clause_max_len)
    return instances
# read_instances("test.txt")

def build_vocabulary(instances: List[Dict],
                     vocab_size: 10000):
    print("\nBuilding Vocabulary.")

    UNK_TOKEN = "@UNK@"
    PAD_TOKEN = "@PAD@"
    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    words = []
    for instance in instances:
        words.extend(instance["text_tokens"])
    token_counts = dict(Counter(words).most_common(vocab_size))
    for token, _ in token_counts.items():
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
        if len(token_to_id) == vocab_size:
            break

    id_to_token = dict(zip(token_to_id.values(), token_to_id.keys()))
    # print(token_to_id)
    # print(id_to_token)
    return (token_to_id, id_to_token)

#build_vocabulary(read_instances("test.txt"), 10000)

def save_vocabulary(vocab_id_to_token: Dict[int, str], vocabulary_path: str) -> None:
    """
    Saves vocabulary to vocabulary_path.
    """
    with open(vocabulary_path, "w") as file:
        # line number is the index of the token
        for idx in range(len(vocab_id_to_token)):
            file.write(vocab_id_to_token[idx] + "\n")

def load_vocabulary(vocabulary_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Loads vocabulary from vocabulary_path.
    """
    vocab_id_to_token = {}
    vocab_token_to_id = {}
    with open(vocabulary_path, "r") as file:
        for index, token in enumerate(file):
            token = token.strip()
            if not token:
                continue
            vocab_id_to_token[index] = token
            vocab_token_to_id[token] = index
    return (vocab_token_to_id, vocab_id_to_token)

def load_glove_embeddings(embeddings_txt_file: str,
                          embedding_dim: int,
                          vocab_id_to_token: Dict[int, str]) -> np.ndarray:
    """
    Given a vocabulary (mapping from index to token), this function builds
    an embedding matrix of vocabulary size in which ith row vector is an
    entry from pretrained embeddings (loaded from embeddings_txt_file).
    """
    tokens_to_keep = set(vocab_id_to_token.values())
    vocab_size = len(vocab_id_to_token)

    embeddings = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file) as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            if not token in tokens_to_keep:
                continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
                continue
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector

    # Estimate mean and std variation in embeddings and initialize it random normally with it
    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    embedding_matrix = np.random.normal(embeddings_mean, embeddings_std,
                                        (vocab_size, embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    for idx, token in vocab_id_to_token.items():
        if token in embeddings:
            embedding_matrix[idx] = embeddings[token]

    return embedding_matrix

def index_instances(instances: List[Dict], token_to_id: Dict) -> List[Dict]:
    """
    Uses the vocabulary to index the fields of the instances. This function
    prepares the instances to be tensorized.
    """
    for instance in instances:
        token_ids = []
        for token in instance["text_tokens"]:
            if token in token_to_id:
                token_ids.append(token_to_id[token])
            else:
                token_ids.append(0) # 0 is index for UNK
        instance["text_tokens_ids"] = token_ids
        instance.pop("text_tokens")
    return instances

def generate_batches(instances: List[Dict], batch_size) -> List[Dict[str, np.ndarray]]:
    """
    Generates and returns batch of tensorized instances in a chunk of batch_size.
    """

    def chunk(items: List[Any], num: int):
        return [items[index:index+num] for index in range(0, len(items), num)]
    batches_of_instances = chunk(instances, batch_size)

    batches = []
    for batch_of_instances in tqdm(batches_of_instances):

        num_token_ids = [len(instance["text_tokens_ids"])
                         for instance in batch_of_instances]
        max_num_token_ids = max(num_token_ids)

        count = min(batch_size, len(batch_of_instances))
        batch = {"inputs": np.zeros((count, max_num_token_ids), dtype=np.int32)}
        # if "labels" in batch_of_instances[0]:
        #     batch["labels"] = np.zeros(count, dtype=np.int32)

        for batch_index, instance in enumerate(batch_of_instances):
            num_tokens = len(instance["text_tokens_ids"])
            inputs = np.array(instance["text_tokens_ids"])
            batch["inputs"][batch_index][:num_tokens] = inputs

            # if "labels" in instance:
            #     batch["labels"][batch_index] = np.array(instance["labels"])
        batches.append(batch)

    return batches

def get_sentence(index: List[int], id_to_token: Dict) -> str:
    # print("\nOutput sentence....")
    str = ""
    for idx in index:
        str = str + " " + id_to_token[idx]
    str = str.strip() + "."
    return str
