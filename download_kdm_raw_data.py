import csv
import os
from datetime import datetime

from dotenv import dotenv_values, load_dotenv
import mysql.connector


BATCH_SIZE = 100000
OUTPUT_DIR = "source_kdm_all"

# Testing controls
# - Set TESTING=True to stop after TEST_MAX_BATCHES batches per table.
TESTING = False
TEST_MAX_BATCHES = 1


MARKET_BUSINESS_BASE_QUERY = """
SELECT
	mb.id,
	mb.name,
	mb.status,
	mb.address,
	mb.description,
	mb.sector,
	mb.note,
	mb.latitude,
	mb.longitude,
	mb.market_id,
	mb.user_id,

	r.long_code  AS regency_id,
	r.name       AS regency_name,

	sd.long_code AS subdistrict_id,
	sd.name      AS subdistrict_name,

	v.long_code  AS village_id,
	v.name       AS village_name,

	s.long_code  AS sls_id,
	s.name       AS sls_name,

	mb.created_at,
	mb.updated_at,
	mb.deleted_at,
	mb.matched_at,
	mb.checked_at,
	mb.duplicate_scan_at,
	mb.match_level,
	mb.upload_id,
	mb.is_locked

FROM market_business mb
LEFT JOIN regencies r     ON r.id = mb.regency_id
LEFT JOIN subdistricts sd ON sd.id = mb.subdistrict_id
LEFT JOIN villages v      ON v.id = mb.village_id
LEFT JOIN sls s           ON s.id = mb.sls_id
ORDER BY mb.id
LIMIT %s OFFSET %s
"""


SUPPLEMENT_BUSINESS_BASE_QUERY = """
SELECT
	sp.id,
	sp.name,
	sp.status,
	sp.address,
	sp.description,
	sp.sector,
	sp.note,
	sp.latitude,
	sp.longitude,

	r.long_code  AS regency_id,
	r.name       AS regency_name,

	sd.long_code AS subdistrict_id,
	sd.name      AS subdistrict_name,

	v.long_code  AS village_id,
	v.name       AS village_name,

	s.long_code  AS sls_id,
	s.name       AS sls_name,

	sp.organization_id,
	sp.user_id,
	sp.upload_id,
	sp.project_id,
	sp.owner,

	sp.created_at,
	sp.updated_at,
	sp.deleted_at,
	sp.matched_at,
	sp.checked_at,
	sp.duplicate_scan_at,
	sp.match_level,
	sp.is_locked

FROM supplement_business sp
LEFT JOIN regencies r     ON r.id = sp.regency_id
LEFT JOIN subdistricts sd ON sd.id = sp.subdistrict_id
LEFT JOIN villages v      ON v.id = sp.village_id
LEFT JOIN sls s           ON s.id = sp.sls_id
ORDER BY sp.id
LIMIT %s OFFSET %s
"""


def load_env() -> dict:
	"""Load environment variables from .env in current directory."""
	load_dotenv(dotenv_path=".env")
	env = dotenv_values(".env")
	if not env:
		raise FileNotFoundError(".env file not found or empty in current directory")
	return env


def get_env_value(env: dict, keys: list[str], required: bool = True, default=None):
	for key in keys:
		value = env.get(key)
		if value is not None and str(value).strip() != "":
			return value
	if required:
		available_keys = ", ".join(sorted([k for k in env.keys()]))
		raise ValueError(
			"Missing environment variable. Expected one of: "
			f"{', '.join(keys)}. Available keys in .env: {available_keys}"
		)
	return default


def build_db_config(env: dict) -> dict:
	host = get_env_value(env, ["DB_HOST", "MYSQL_HOST", "DATABASE_HOST"])
	port = int(get_env_value(env, ["DB_PORT", "MYSQL_PORT", "DATABASE_PORT"], required=False, default=3306))
	user = get_env_value(env, ["DB_USER", "MYSQL_USER", "DATABASE_USER", "DB_USERNAME", "MYSQL_USERNAME"])
	password = get_env_value(env, ["DB_PASSWORD", "MYSQL_PASSWORD", "DATABASE_PASSWORD"], required=False, default="")
	database = get_env_value(
		env,
		[
			"DB_NAME",
			"DB_DATABASE",
			"MYSQL_DATABASE",
			"MYSQL_DB",
			"DATABASE_NAME",
			"DATABASE",
		],
	)

	return {
		"host": host,
		"port": port,
		"user": user,
		"password": password,
		"database": database,
	}


def export_table_in_batches(connection, query: str, output_file: str, batch_size: int, max_batches: int | None = None):
	offset = 0
	total_rows = 0
	header_written = False
	batches = 0
	columns: list[str] = []

	with open(output_file, mode="w", newline="", encoding="utf-8-sig") as csv_file:
		writer = csv.writer(csv_file)

		while True:
			if max_batches is not None and batches >= max_batches:
				print(f"Reached max_batches={max_batches}. Stopping early (testing mode).")
				break

			with connection.cursor() as cursor:
				cursor.execute(query, (batch_size, offset))
				rows = cursor.fetchall()

				if not rows:
					break

				if not header_written:
					columns = [column[0] for column in cursor.description]
					writer.writerow(columns)
					header_written = True

				writer.writerows(rows)

				fetched = len(rows)
				total_rows += fetched
				offset += fetched
				batches += 1
				print(f"Wrote {fetched} rows (total: {total_rows}) to {output_file}")

				if fetched < batch_size:
					break

	print(f"Finished export: {output_file} ({total_rows} rows, {batches} batches)")
	return {
		"output_file": output_file,
		"rows": total_rows,
		"batches": batches,
		"columns": columns,
	}


def main():
	env = load_env()
	db_config = build_db_config(env)

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

	market_output = os.path.join(OUTPUT_DIR, f"market_business_{timestamp}.csv")
	supplement_output = os.path.join(OUTPUT_DIR, f"supplement_business_{timestamp}.csv")

	max_batches = TEST_MAX_BATCHES if TESTING else None

	print("Connecting to database...")
	with mysql.connector.connect(**db_config) as connection:
		print("Exporting market_business...")
		market_summary = export_table_in_batches(
			connection,
			MARKET_BUSINESS_BASE_QUERY,
			market_output,
			BATCH_SIZE,
			max_batches=max_batches,
		)

		print("Exporting supplement_business...")
		supplement_summary = export_table_in_batches(
			connection,
			SUPPLEMENT_BUSINESS_BASE_QUERY,
			supplement_output,
			BATCH_SIZE,
			max_batches=max_batches,
		)

	print("\nSummary")
	print(f"- market_business:   {market_summary['rows']} rows, {market_summary['batches']} batches -> {market_summary['output_file']}")
	print(f"- supplement_business: {supplement_summary['rows']} rows, {supplement_summary['batches']} batches -> {supplement_summary['output_file']}")

	print("All exports completed")


if __name__ == "__main__":
	main()
