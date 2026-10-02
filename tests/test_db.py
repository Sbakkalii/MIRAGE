from pymilvus import MilvusClient

client = MilvusClient("../rag_v1_milvus.db")
collection_name = "rag_v1"

res = client.list_collections()
print(res)