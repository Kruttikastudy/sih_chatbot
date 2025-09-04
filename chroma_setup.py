import chromadb
from chromadb.utils import embedding_functions

# ----------------------------
# Initialize Chroma persistent client
# ----------------------------
client = chromadb.PersistentClient(path="D:/Hackathons/SIH/Prototype/chroma_db")

# ----------------------------
# Embedding function (Sentence-Transformers)
# ----------------------------
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"   # fast, accurate, small footprint
)

# ----------------------------
# Create or get collection
# ----------------------------
collection = client.get_or_create_collection(
    name="db_schema",
    embedding_function=embed_fn
)

# ----------------------------
# Schema description for Argo table
# ----------------------------
schema_description = {
    "argo": [
        {"column": "id", "type": "integer", "description": "Unique identifier for each Argo float measurement row."},
        {"column": "CYCLE_NUMBER", "type": "text", "description": "The cycle number of the Argo float; indicates the sequential measurement cycle."},
        {"column": "DATA_MODE", "type": "text", "description": "Mode of the data: typically 'R' for real-time or 'D' for delayed-mode processed data."},
        {"column": "DIRECTION", "type": "text", "description": "Indicates whether the profile measurement is ascending ('A') or descending ('D') through the water column."},
        {"column": "PLATFORM_NUMBER", "type": "text", "description": "Unique identifier for the Argo float platform that collected the data."},
        {"column": "POSITION_QC", "type": "text", "description": "Quality control flag for the GPS position; indicates reliability of latitude and longitude."},
        {"column": "PRES", "type": "text", "description": "Measured water pressure in decibars; represents depth of measurement."},
        {"column": "PRES_ERROR", "type": "text", "description": "Estimated error or uncertainty in the pressure measurement."},
        {"column": "PRES_QC", "type": "text", "description": "Quality control flag for the pressure data; indicates if the measurement passed QC checks."},
        {"column": "PSAL", "type": "text", "description": "Measured salinity in Practical Salinity Units (PSU) at the given depth and location."},
        {"column": "PSAL_ERROR", "type": "text", "description": "Estimated error or uncertainty in the salinity measurement."},
        {"column": "PSAL_QC", "type": "text", "description": "Quality control flag for salinity measurements; indicates reliability of the value."},
        {"column": "TEMP", "type": "text", "description": "Measured water temperature in degrees Celsius at the given depth and location."},
        {"column": "TEMP_ERROR", "type": "text", "description": "Estimated error or uncertainty in the temperature measurement."},
        {"column": "TEMP_QC", "type": "text", "description": "Quality control flag for temperature measurements; indicates reliability of the value."},
        {"column": "TIME", "type": "text", "description": "Timestamp of the measurement in UTC; indicates when the data was collected."},
        {"column": "TIME_QC", "type": "text", "description": "Quality control flag for the timestamp; indicates if the recorded time is valid."},
        {"column": "LATITUDE", "type": "text", "description": "Latitude of the measurement in decimal degrees; north is positive."},
        {"column": "LONGITUDE", "type": "text", "description": "Longitude of the measurement in decimal degrees; east is positive."}
    ]
}

# ----------------------------
# Prepare vectors for ChromaDB
# ----------------------------
ids = []
documents = []
metadatas = []

for table_name, columns in schema_description.items():
    for col in columns:
        ids.append(f"{table_name}.{col['column']}")
        documents.append(f"Column: {col['column']}, Type: {col['type']}, Description: {col['description']}")
        metadatas.append({
            "table": table_name,
            "column": col['column'],
            "type": col['type']
        })

# ----------------------------
# Add schema embeddings to ChromaDB
# ----------------------------
collection.add(documents=documents, metadatas=metadatas, ids=ids)

print("✅ Argo schema successfully stored in ChromaDB with Sentence-Transformers embeddings!")
print("📂 ChromaDB persisted at: D:/Hackathons/SIH/Prototype/chroma_db")
