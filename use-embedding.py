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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai import common

parser = argparse.ArgumentParser()
parser.add_argument('--model-name', type=str, help='Model name or local path containing the model, must be the same as the the model used to build the embeddings', required=True)
parser.add_argument('--embeddings-db-path', type=str, help='Duckdb path of the emebddings database', required=True)
parser.add_argument('--function-db-path', type=str, help='Duckdb path of the function database. If not provided, function source code will not be printed', required=False)
parser.add_argument('--max-length', type=int, help='Maximum length of the input text, longer text will be truncated', default=1024)
parser.add_argument('--limit', type=int, help='Number of top matches to return', default=5)
parser.add_argument('query', type=str, help='Query string to search for similar documents')

args = parser.parse_args()



model_path = args.model_name
model_name = model_path.split("/")[-1]
embedding_size = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

print(f'Query embeddings using model {model_name}')

def cal_embedding(text,  max_length=args.max_length, truncation=True):
    tokens = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=truncation)
    with torch.no_grad():
        return model(**tokens).last_hidden_state.mean(dim=1).numpy().flatten().tolist()

em = cal_embedding(args.query)
embeddings_con = duckdb.connect(database=args.embeddings_db_path)

functions_con = duckdb.connect(database=args.function_db_path) if args.function_db_path else None

# Search for the top matches
query_sql = f"SELECT code_hash FROM embeddings ORDER BY array_distance(vec, ?::FLOAT[{embedding_size}]) LIMIT ?;"
results = embeddings_con.execute(query_sql, [em, args.limit]).fetchall()

print(f'Top {args.limit} matches for query "{args.query}":')
for r in results:
    print(f'Code hash: {r[0]}')
    if not functions_con:
        continue
    if functions_con:
        function_ids = embeddings_con.execute(f"SELECT function_id FROM function_code_hash WHERE code_hash = ?", [r[0]]).fetchall()
        print('Function ids:', function_ids)
        fid = function_ids[0][0]
        function = functions_con.execute(f"SELECT source_code FROM function WHERE id = ?", [fid]).fetchall()
        print('Function source code:', function[0][0])
