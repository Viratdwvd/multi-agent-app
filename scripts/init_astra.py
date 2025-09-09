from __future__ import annotations
import os
from astrapy.db import AstraDB

endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
app_token = os.getenv("ASTRA_DB_APP_TOKEN")
keyspace = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
collection = os.getenv("ASTRA_COLLECTION", "docs")

assert endpoint and app_token, "Set ASTRA_DB_API_ENDPOINT and ASTRA_DB_APP_TOKEN"

adb = AstraDB(token=app_token, api_endpoint=endpoint, namespace=keyspace)

print("Ensuring collection exists…")
resp = adb.create_collection_if_not_exists(
    collection_name=collection,
    dimension=384,   # Adjust to your embedding model
    metric="cosine",
    indexing="dense_vector",
)
print(resp)

print("Collections:", [c["name"] for c in adb.list_collections()])
