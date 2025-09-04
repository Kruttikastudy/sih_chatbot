import chromadb

# ----------------------------
# Load the persistent client
# ----------------------------
client = chromadb.PersistentClient(path="D:/Hackathons/SIH/Prototype/chroma_db")

# ----------------------------
# Get your collection
# ----------------------------
collection = client.get_collection("db_schema")

# ----------------------------
# Basic info
# ----------------------------
print("📊 Total entries in collection:", collection.count())

# Peek into stored records
print("\n🔎 Sample records:")
print(collection.peek())   # shows a few documents, metadatas, ids

# ----------------------------
# Test a semantic query
# ----------------------------
query = "Find the column related to salinity"
results = collection.query(
    query_texts=[query],
    n_results=3   # top 3 most relevant results
)

print("\n🧠 Query:", query)
print("✅ Top matches:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"- {doc}")
    print(f"  ↳ Metadata: {meta}")
