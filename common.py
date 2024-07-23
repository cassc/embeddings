import hashlib

def sha256(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode('utf-8'))
    sha256_hex_digest = sha256_hash.hexdigest()
    return sha256_hex_digest
