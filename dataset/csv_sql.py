import pandas as pd

# Use a raw string for Windows paths to avoid escape sequence issues
file_path = r"g:\project 2\Watchbird\dataset\bird_tracking.csv"

# Read CSV
df = pd.read_csv(file_path)

# Rename columns to match your database
df_sql = df.rename(columns={
    "date_time": "Timestamp",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "altitude": "Altitude"
})

# Add Device_id column with value 2
df_sql["Device_id"] = 2

# Keep only the columns needed for SQL
df_sql = df_sql[["Timestamp", "Device_id", "Latitude", "Longitude", "Altitude"]]

# Generate SQL INSERT statements
sql_statements = []
for index, row in df_sql.iterrows():
    sql = f'INSERT INTO "Data" ("Timestamp", "Device_id", "Latitude", "Longitude", "Altitude") ' \
          f'VALUES (\'{row["Timestamp"]}\', {row["Device_id"]}, {row["Latitude"]}, {row["Longitude"]}, {row["Altitude"]});'
    sql_statements.append(sql)

# Write SQL statements to a file
output_file = r"g:\project 2\Watchbird\dataset\insert_data.sql"
with open(output_file, "w") as f:
    f.write("\n".join(sql_statements))

print(f"Generated {len(sql_statements)} SQL statements in {output_file}")
