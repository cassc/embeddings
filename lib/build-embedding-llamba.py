from typing import List
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
from transformers import LlamaTokenizer, LlamaModel
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import os
from chromadb.types import Metadata
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib import common

# parser = argparse.ArgumentParser()
# parser.add_argument('--model-name', type=str, help='Model name or local path containing the model, must be the same as the the model used to build the embeddings', required=True)
# parser.add_argument('--embeddings-db-path', type=str, help='Duckdb path of the emebddings database', required=True)
# parser.add_argument('--function-db-path', type=str, help='Duckdb path of the function database. If not provided, function source code will not be printed', required=False)
# parser.add_argument('--max-length', type=int, help='Maximum length of the input text, longer text will be truncated', default=1024)

# args = parser.parse_args()

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("/data-ssd/chen/llama-models/models/Meta-Llama-3.1-70B")
model = LlamaModel.from_pretrained("/data-ssd/chen/llama-models/models/Meta-Llama-3.1-70B-Instruct")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = tokenizer("Your input text here", return_tensors="pt")
outputs = model(**inputs)

# Get the embeddings from the hidden states
embeddings = outputs.last_hidden_state.mean(dim=1)  # Example using mean pooling

print(embeddings)
