import os
import re
import streamlit as st
from sqlalchemy import create_engine, text
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ------------------ Groq API Setup ------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError("⚠ Please set GROQ_API_KEY environment variable. Get it from https://console.groq.com/keys")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

MODEL_NAME = "qwen/qwen3-32b"  # More stable for SQL generation

# ------------------ SQL Helpers ------------------
def extract_sql_from_response(response: str) -> str:
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'```', '', response)
    response = re.sub(r'(?i)sql', '', response)
    lines = response.strip().split('\n')
    sql_lines = []
    found_sql = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(keyword in line.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE']):
            found_sql = True
        if found_sql:
            sql_lines.append(line)
    if sql_lines:
        return re.sub(r'\s+', ' ', ' '.join(sql_lines)).strip()
    return response.strip()

def sanitize_sql(sql: str) -> str:
    if "EXTRACT(" in sql.upper():
        sql = re.sub(
            r"EXTRACT\(HOUR FROM TIME\).?20.?EXTRACT\(MINUTE FROM TIME\).*?BETWEEN 40 AND 50",
            "CAST(TIME AS TIMESTAMP) BETWEEN '2002-03-16 20:40:00' AND '2002-03-16 20:50:00'",
            sql,
            flags=re.IGNORECASE,
        )
    return sql

def generate_simple_sql(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "salinity" in prompt_lower and "temperature" in prompt_lower:
        return "SELECT PSAL, TEMP, PLATFORM_NUMBER, TIME, LATITUDE, LONGITUDE FROM argo WHERE PLATFORM_NUMBER = '2900210' LIMIT 20;"
    elif "platform" in prompt_lower and any(char.isdigit() for char in prompt):
        platform_match = re.search(r'\b(\d{7})\b', prompt)
        if platform_match:
            platform_num = platform_match.group(1)
            return f"SELECT * FROM argo WHERE PLATFORM_NUMBER = '{platform_num}' LIMIT 20;"
    if any(kw in prompt_lower for kw in ["select", "show", "get", "find"]):
        return "SELECT * FROM argo LIMIT 10;"
    elif "count" in prompt_lower:
        return "SELECT COUNT(*) FROM argo;"
    return "-- Unable to generate SQL query for this request. Please be more specific."

def query_llm(prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        """
                        "DO NOT PRINT YOUR THINKING"
                        "Whatever you think should NOT be printed."
                        "Do not include explanations, thinking process, or markdown formatting. "
                        "You are a PostgreSQL expert. "
                        "Generate ONLY executable SQL queries. "
                        "The database contains a single table named argo with the following columns:"
                        "- CYCLE_NUMBER"
                        "- DATA_MODE"
                        "- DIRECTION"
                        "- PLATFORM_NUMBER"
                        "- POSITION_QC"
                        "- PRES"
                        "- PRES_ERROR"
                        "- PRES_QC"
                        "- PSAL"
                        "- PSAL_ERROR"
                        "- PSAL_QC"
                        "- TEMP"
                        "- TEMP_ERROR"
                        "- TEMP_QC"
                        "- TIME_QC"
                        "- LATITUDE (stored as text, must be cast with CAST(LATITUDE AS DOUBLE PRECISION))"
                        "- LONGITUDE (stored as text, must be cast with CAST(LONGITUDE AS DOUBLE PRECISION))"
                        "- TIME (stored as text, must be cast with CAST(TIME AS TIMESTAMP))"
                        "Return only the SQL query that can be directly executed, using the necessary CAST for LATITUDE, LONGITUDE, and TIME."
                        """
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            top_p=1,
            stream=False
        )
        raw_response = completion.choices[0].message.content.strip()
        sql = extract_sql_from_response(raw_response)
        return sanitize_sql(sql)
    except Exception as e:
        st.warning(f"⚠ Groq API Error: {e}")
        return generate_simple_sql(prompt)

def generate_natural_response(user_query: str, sql_query: str, results: list, row_count: int) -> str:
    try:
        results_preview = results[:5] if results else []
        results_text = ""
        if results_preview:
            for i, row in enumerate(results_preview, 1):
                row_dict = dict(row._mapping) if hasattr(row, '_mapping') else row
                results_text += f"Row {i}: {row_dict}\n"
        if len(results) > 5:
            results_text += f"... and {len(results) - 5} more rows\n"
        prompt = f"""
        You are an expert oceanographer and data analyst specializing in Argo float data. 
        USER QUESTION: {user_query}
        SQL QUERY EXECUTED: {sql_query}
        QUERY RESULTS ({row_count} total rows):
        {results_text if results_text else "No data found"}
        INSTRUCTIONS:
          "DO NOT PRINT YOUR THINKING"
          "Whatever you think should NOT be printed."
        1. Provide a natural language summary of the results
        2. Answer the user's original question based on the data
        3. Include relevant insights, patterns, or observations
        4. Mention data quality indicators if present (QC flags, errors)
        """
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful oceanographic data analyst. Provide clear, insightful responses about Argo float data."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            top_p=1,
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"⚠ Error generating natural language response: {e}")
        return "Error generating response."

# ------------------ Postgres Setup ------------------
PG_URI = "postgresql+psycopg2://postgres:Jhaveri%401117@localhost:5432/floatchat_db"
engine = create_engine(PG_URI)

def fetch_sample(engine):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
        )
        return [row[0] for row in result]

def get_table_schema(engine, table_name):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = :table_name AND table_schema = 'public'
            ORDER BY ordinal_position
            """),
            {"table_name": table_name}
        )
        return [{"column": row[0], "type": row[1], "nullable": row[2]} for row in result]

# ------------------ Chroma Setup ------------------
def get_chroma_client():
    return chromadb.PersistentClient(path=r"D:\Hackathons\SIH\Prototype\chroma_db")

def fetch_from_chroma(client, collection_name, query):
    try:
        col = client.get_collection(collection_name)
        results = col.query(query_texts=[query], n_results=5)
        return results
    except Exception as e:
        st.warning(f"⚠ Chroma error: {e}")
        return None

def format_chroma_context(chroma_results):
    if not chroma_results or not chroma_results.get('documents'):
        return "No relevant schema information found in ChromaDB."
    context_parts = []
    documents = chroma_results['documents'][0] if chroma_results['documents'] else []
    metadatas = chroma_results['metadatas'][0] if chroma_results.get('metadatas') else []
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        context_parts.append(f"Schema Info: {doc}")
        if metadata:
            context_parts.append(f"Metadata: {metadata}")
    return "\n".join(context_parts)

# ------------------ MCP Class ------------------
class MCP:
    def __init__(self):
        self.pg = engine
        self.chroma = get_chroma_client()
        self.history = []

    def process_query(self, user_query: str):
        chroma_results = None
        possible_collections = ["db_schema"]
        for collection_name in possible_collections:
            chroma_results = fetch_from_chroma(self.chroma, collection_name, user_query)
            if chroma_results and chroma_results.get('documents'):
                break
        chroma_context = format_chroma_context(chroma_results)

        pg_tables = fetch_sample(self.pg)
        schema_info = {}
        target_tables = ["argo"] if "argo" in pg_tables else pg_tables[:2]
        for table in target_tables:
            schema_info[table] = get_table_schema(self.pg, table)

        focused_prompt = "Previous queries and SQL (for context):\n"
        for past in self.history[-5:]:
            focused_prompt += f"USER: {past['query']}\nSQL: {past['sql']}\n\n"

        focused_prompt += f"Current query: {user_query}\n=== DATABASE TABLES AND COLUMNS ===\n"
        for table, schema in schema_info.items():
            focused_prompt += f"\n{table}: "
            col_names = [col['column'] for col in schema if isinstance(schema, dict)]
            focused_prompt += ", ".join(col_names)

        if chroma_results and chroma_results.get('documents'):
            focused_prompt += f"\n\n=== RELEVANT SCHEMA INFO (from ChromaDB) ===\n{chroma_context[:500]}"

        sql = query_llm(focused_prompt)
        self.history.append({"query": user_query, "sql": sql})
        return sql

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="🌊 Argo Data Chatbot", layout="wide")
st.title("🌊 Argo Data Chatbot")
st.write("Ask questions about the Argo float dataset. The system will generate SQL, fetch data, and summarize results.")

# Persist MCP instance across reruns
if "mcp" not in st.session_state:
    st.session_state.mcp = MCP()

mcp = st.session_state.mcp

# Persist chat messages
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []  # Each item: {"role": "user"/"assistant", "content": "..."}

# Response type selection
response_type = st.radio(
    "🎯 Choose output format:",
    options=["Natural language response", "Raw data", "Both"],
    index=0
)

# Chat input
user_input = st.chat_input("💬 Type your question about Argo data:")

if user_input:
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    with st.spinner("🔄 Generating SQL and fetching results..."):
        # Generate SQL
        sql = mcp.process_query(user_input)

        # Execute SQL
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                row_count = len(rows)
        except Exception as e:
            rows, row_count = [], 0
            st.session_state.chat_messages.append({"role": "assistant", "content": f"❌ SQL Execution Error: {e}"})

        # Generate natural language response (if needed)
        natural_response = ""
        if response_type in ["Natural language response", "Both"]:
            try:
                natural_response = generate_natural_response(user_input, sql, rows, row_count)
            except Exception as e:
                natural_response = f"⚠ Error generating response: {e}"

        # Include raw data (if needed)
        raw_data_text = ""
        if response_type in ["Raw data", "Both"] and rows:
            display_rows = [dict(row._mapping) for row in rows[:20]]
            raw_data_text = f"\n\n📊 **Raw Results (first 20 rows shown):**\n{display_rows}"
            if len(rows) > 20:
                raw_data_text += f"\n... and {len(rows) - 20} more rows"

        # Combine into assistant message
        assistant_message = f"📄 **Generated SQL:**\n```sql\n{sql}\n```"
        if natural_response:
            assistant_message += f"\n\n💬 **Response:**\n{natural_response}"
        if raw_data_text:
            assistant_message += raw_data_text

        # Append assistant message
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_message})

# Display chat messages in order
for message in st.session_state.chat_messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])


