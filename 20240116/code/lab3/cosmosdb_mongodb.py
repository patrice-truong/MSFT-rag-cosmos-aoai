def create_index(collection, index_name, dimension_size=1536, num_lists=100):
    # check if index exists
    existing_indexes = collection.index_information()
    if index_name not in existing_indexes:
        # Define the index key, specifying the field "embedding" with a cosine similarity search type
        index_key = [("vectorContent", "cosmosSearch")]
        # Define the options for the index
        index_options = {
            "kind": "vector-ivf",   
            "numLists": num_lists,        # Number of lists (partitions) in the index
            "similarity": "COS",    # Similarity metric: Cosine similarity
            "dimensions": dimension_size      # Number of dimensions in the vectors
        }
        # Create the index with the specified name and options
        index = collection.create_index(index_key, name=index_name, cosmosSearchOptions=index_options)
        return index

def get_docs(collection):
    limit = 10
    docs = []
    for doc in collection.find().limit(limit):
        serialized_doc = {**doc, "id": str(doc["id"])}
        docs.append(serialized_doc)
    return docs

def count_docs(collection):
    c = collection.count_documents({})
    return c

def insert_one(collection, doc):
    return collection.insert_one(doc)

def insert_one_if_not_exists(collection, doc):    
    doc_id = str(doc["id"])
    
    if collection.count_documents({"id": doc_id}) == 0:
        return collection.insert_one(doc)
    else:
        return f"Document with _id {doc_id} already exists, no insertion performed."

def insert_many(collection, docs):
    return collection.insert_many(docs)

def insert_many_if_not_exist(collection, docs):
    existing_doc_ids = [str(doc["_id"]) for doc in collection.find({})]

    new_docs = []
    for doc in docs:
        doc_id = str(doc["id"])
        if doc_id not in existing_doc_ids:
            new_docs.append(doc)

    if new_docs:
        result = collection.insert_many(new_docs)
        return f"Successfully inserted {len(new_docs)} new docs."
    else:
        return "No new docs to insert."


def similar(collection, query_vector, limit=5, min_score=0.8):

    pipeline = [
            {
                '$search': {
                    'cosmosSearch': {
                        'vector': query_vector,
                        'path': 'embedding',
                        'k': limit
                    },
                    'returnStoredSource': True
                }
            }
        ]

    docs = []
    for doc in collection.aggregate(pipeline):
        serialized_doc = {**doc, "id": str(doc["id"])}
        docs.append(serialized_doc)

    return docs
    