# python build-embedding.py --model-name /home/garfield/projects/bigcode/starencoder --embeddings-db-path /home/garfield/tmp/embeddings.duckdb --function-db-path /home/garfield/projects/sbip-sg/blockchain-data/contracts-with-functions-0703.duckdb
from typing import List
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
from chromadb.types import Metadata
import chromadb.utils.embedding_functions as embedding_functions
import hashlib
import numpy as np
import duckdb
import sys
import argparse
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import common

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, help='Model name or local path containing the model, must be the same as the the model used to build the embeddings', required=True)
parser.add_argument('--embeddings-db-path', type=str, help='Duckdb path of the emebddings database', required=True)
parser.add_argument('--function-db-path', type=str, help='Duckdb path of the function database. If not provided, function source code will not be printed', required=False)
parser.add_argument('--max-length', type=int, help='Maximum length of the input text, longer text will be truncated', default=1024)

args = parser.parse_args()


model_path = args.model_name
model_name = model_path.split("/")[-1]
embedding_size = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

print(f'Building embeddings using model {model_name}')

def cal_embedding(text, max_length=args.max_length, truncation=True):
    tokens = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=truncation)
    with torch.no_grad():
        return model(**tokens).last_hidden_state.mean(dim=1).numpy().flatten().tolist()

function_con = duckdb.connect(database=args.function_db_path)

embeddings_con = duckdb.connect(database=args.embeddings_db_path)

try:
    embeddings_con.execute(f'''
    INSTALL vss;
    LOAD vss;
    SET hnsw_enable_experimental_persistence = true;
    CREATE TABLE function_code_hash ( function_id STRING primary key, code_hash STRING );
    CREATE TABLE embeddings ( code_hash STRING primary key, vec FLOAT[{embedding_size}] );
    CREATE INDEX cos_idx ON embeddings USING HNSW (vec) WITH (metric = 'cosine');
    ''')
except Exception as e:
    if 'already exists!' in str(e):
        print('Assuming table already exists, ignoring', e)
    else:
        raise e

results = function_con.execute("SELECT id, contract_id, source_code FROM 'functions' where source_code!=?", [""]).fetchall()

def check_function_processed(id):
    return embeddings_con.execute("SELECT function_id FROM function_code_hash WHERE function_id=? limit 1", [id]).fetchone() is not None

# from code hash to embedding
code_hash_seen = set()

for r in tqdm(results, desc="Calculating embeddings", ncols=80):
    id, contract_id, source_code = r
    if check_function_processed(id):
        continue

    code_hash = common.sha256(source_code)
    if code_hash not in code_hash_seen:
        em = cal_embedding(source_code)
        code_hash_seen.add(code_hash)
        embeddings_con.execute("INSERT OR IGNORE INTO embeddings VALUES (?, ?)", [code_hash, em])

    embeddings_con.execute("INSERT OR IGNORE INTO function_code_hash VALUES (?, ?)", [id, code_hash]);
