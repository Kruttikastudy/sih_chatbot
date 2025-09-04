from sqlalchemy import create_engine, text
import pandas as pd
import os

# Database connection details
db_user = "postgres"
db_password = os.getenv("DB_PASSWORD")
db_host = "localhost"
db_port = "5432"
db_name = "floatchat_db"
table_name = "argo"

# Path to merged CSV
csv_file_path = r"D:\Hackathons\SIH\Prototype\Argo_Arabian_Sea_Strips(50,70,-60,20) - Argo_Arabian_Sea_Strips(50,70,-60,20).csv"

# Number of rows already inserted
already_done = 0  # change as needed

try:
    # Read CSV with header
    df = pd.read_csv(csv_file_path, skiprows=already_done)

    # Show columns
    print("Columns in CSV:", df.columns.tolist())

    # Create PostgreSQL connection
    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    with engine.begin() as conn:
        # Create table dynamically based on CSV columns
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {', '.join([f'"{col}" TEXT' for col in df.columns])}
        );
        """
        conn.execute(text(create_table_query))

        # Drop the unwanted 'Unnamed: 0' column if it exists
        drop_column_query = f"""
        ALTER TABLE {table_name}
        DROP COLUMN IF EXISTS "Unnamed: 0";
        """
        conn.execute(text(drop_column_query))

    # Insert all rows at once (pandas will ignore missing columns)
    # Drop 'Unnamed: 0' from dataframe before insertion
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Inserted {len(df)} new rows starting from row {already_done + 1}.")

except Exception as e:
    print(f"An error occurred: {e}")
