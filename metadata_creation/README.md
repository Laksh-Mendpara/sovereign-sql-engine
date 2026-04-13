# Metadata Creation

This utility reads table and column metadata from a dataset folder of CSV files and loads that metadata into a SQLite Cloud database.

It creates two tables:

- `table_metadata`
- `column_metadata`

The script expects a metadata directory where:

- each table has its own `*.csv` file describing columns
- one shared `table_metadata_file.csv` contains table-level descriptions

## What The Script Imports

For each table CSV in the input directory, the script:

1. creates or refreshes the metadata tables in SQLite Cloud
2. reads table descriptions from `table_metadata_file.csv`
3. inserts one row into `table_metadata` per table
4. inserts one row into `column_metadata` per column definition

The script skips `table_metadata_file.csv` from the per-table import loop and uses it only as the source for table descriptions.

## Expected Directory Structure

The directory you pass to the script should look like this:

```text
database_description/
├── circuits.csv
├── drivers.csv
├── races.csv
├── ...
└── table_metadata_file.csv
```

## Expected CSV Formats

### 1. Table metadata file

`table_metadata_file.csv` should contain:

```csv
table_name,description
circuits,Information about Formula 1 racing circuits including location, country, and geographical coordinates.
```

### 2. Column metadata files

Each table CSV should contain these headers:

```csv
original_column_name,column_name,column_description,data_format,value_description
```

Example:

```csv
original_column_name,column_name,column_description,data_format,value_description
circuitId,circuit Id,unique identification number of the circuit,integer,
```

## Environment Variables

Create a `.env` file in this directory using `.env.example` as a reference.

Required variables:

```env
SQLITE_HOST=<your_key_here>
SQLITE_PORT=<your_key_here>
SQLITE_DB=<your_key_here>
SQLITE_API_KEY=<your_key_here>
```

These are used to build the SQLite Cloud connection string.

## Setup

This project uses `uv`.

From [metadata_creation](/home/laksh/repos/sovereign-sql-engine/metadata_creation):

```bash
uv sync
cp .env.example .env
```

Then fill in your real SQLite Cloud credentials in `.env`.

## Usage

Run the script by passing the metadata directory path as the first argument:

```bash
uv run python main.py /path/to/database_description
```

Example:

```bash
uv run python main.py /home/laksh/Downloads/data_minidev/MINIDEV/dev_databases/formula_1/database_description
```

If you are already inside the project virtual environment, you can also run:

```bash
python main.py /path/to/database_description
```

Set `LOG_LEVEL=DEBUG` before running the script if you want more verbose import and Pinecone indexing logs.

## Output

On success, the script prints:

```text
Successfully indexed <n> tables.
```

Example:

```text
Successfully indexed 13 tables.
```

## Database Tables Created

### `table_metadata`

Columns:

- `id`
- `table_name`
- `description`

### `column_metadata`

Columns:

- `id`
- `table_id`
- `original_name`
- `friendly_name`
- `description`
- `data_format`
- `value_description`

## Notes

- The script drops and recreates `table_metadata` and `column_metadata` on every run.
- CSV files are read with encoding fallbacks: `utf-8-sig`, `cp1252`, then `latin-1`.
- Header names and values are trimmed to handle trailing spaces in input CSVs.

## Troubleshooting

### Missing command-line argument

If you run the script without a directory path, it exits with:

```text
Usage: python main.py <csv_directory>
```

### Missing `table_metadata_file.csv`

Make sure the directory you pass contains `table_metadata_file.csv`. The script expects it at:

```text
<csv_directory>/table_metadata_file.csv
```

### SQLite Cloud connection errors

Check that:

- `.env` exists in [metadata_creation](/home/laksh/repos/sovereign-sql-engine/metadata_creation)
- `SQLITE_HOST` is correct
- `SQLITE_PORT` is correct
- `SQLITE_DB` is correct
- `SQLITE_API_KEY` is correct

### Dependency issues

If modules are missing, install dependencies again:

```bash
uv sync
```
