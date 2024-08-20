import boto3
import time
import logging
import torch
from transformers import  AutoTokenizer, AutoModel
import os
import duckdb
import sys
import json
from dotenv import load_dotenv
load_dotenv('.envrc')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sqs = boto3.client('sqs', region_name=os.getenv('AWS_REGION'))
sqs_task_queue_url = os.getenv('SQS_TASK_QUEUE_URL')
sqs_result_queue_url = os.getenv('SQS_RESULT_QUEUE_URL')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

model_path =  os.getenv('MODEL_NAME', '')
model_name = model_path.split("/")[-1]
embedding_size = int(os.getenv('EMBEDDING_SIZE', '768'))
max_length = int(os.getenv('MAX_LENGTH', '1024'))
embeddings_db_path=os.getenv('EMBEDDINGS_DB_PATH')
function_db_path=os.getenv('FUNCTION_DB_PATH')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

embeddings_con = duckdb.connect(database=embeddings_db_path)
functions_con = duckdb.connect(database=function_db_path)

print(f'Query embeddings using model {model_name}')

def cal_embedding(text,  max_length=max_length, truncation=True):
    tokens = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=truncation)
    with torch.no_grad():
        return model(**tokens).last_hidden_state.mean(dim=1).numpy().flatten().tolist()


def process_message(message):
    """
    Process the SQS message.
    This function should contain the logic to process the message.
    """
    logger.info(f"Processing message: {message}")
    body = message['Body']
    body = json.loads(body)
    peer_id = body['peer_id']
    query = body['data']['payload']
    id = body['data']['id']
    limit = body['data'].get('limit', 10)

    em = cal_embedding(query)

    # Search for the top matches
    query_sql = f"SELECT code_hash FROM embeddings ORDER BY array_distance(vec, ?::FLOAT[{embedding_size}]) LIMIT ?;"
    results = embeddings_con.execute(query_sql, [em, limit]).fetchall()
    found = []

    for r in results:
        function_ids = embeddings_con.execute(f"SELECT function_id FROM function_code_hash WHERE code_hash = ?", [r[0]]).fetchall()
        fid = function_ids[0][0]
        function = functions_con.execute(f"SELECT source_code FROM functions WHERE id = ?", [fid]).fetchall()[0]
        if not function:
            logger.warn('Function not found with id:', dict(fid=fid, code_hash=r[0]))
        found.append(function[0]) # get the source_code only

    msg = dict(peer_id=peer_id, result=json.dumps(dict(found=found, id=id)))
    publish_message(json.dumps(msg))


def publish_message(msg: str):
    """
    Publish the processed result message to the result queue
    """
    try:
        response = sqs.send_message(
            QueueUrl=sqs_result_queue_url,
            MessageBody=msg
        )
        return response
    except Exception as e:
        logger.error(f"Failed to publish message to queue: {e}")
        return None




def delete_message(receipt_handle):
    """
    Delete the processed message from the queue.
    """
    try:
        sqs.delete_message(
            QueueUrl=sqs_task_queue_url,
            ReceiptHandle=receipt_handle
        )
        logger.info("Message deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting message: {e}")

def poll_queue():
    """
    Poll the SQS queue for messages and process them.
    """
    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=sqs_task_queue_url,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=20,
                VisibilityTimeout=30
            )
            messages = response.get('Messages', [])
            if not messages:
                logger.info("No messages received.")
                continue

            for message in messages:
                try:
                    process_message(message)
                except Exception as e:
                    logger.error('Process messasge error', message, stack_info=True, exc_info=e)
                finally:
                    delete_message(message['ReceiptHandle'])


        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
            time.sleep(5)  # Wait for 5 seconds before retrying

if __name__ == "__main__":
    logger.info("Starting SQS message polling...")
    poll_queue()
