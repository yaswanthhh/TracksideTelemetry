from pathlib import Path
import csv

RAW_DIR = Path("data/raw")
DELIMITERS = [",", ";", "\t", "|"]


def detect_delimiter(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8-sig", errors="replace") as file:
        sample = file.read(8192)

    try:
        return csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except csv.Error:
        header = sample.splitlines()[0]
        return max(DELIMITERS, key=header.count)


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files found. Put MotoGP 18 CSV files in data/raw first."
        )

    print(f"Files found: {len(files)}\n")

    for csv_path in files:
        delimiter = detect_delimiter(csv_path)

        with csv_path.open("r", encoding="utf-8-sig", errors="replace") as file:
            reader = csv.reader(file, delimiter=delimiter)
            header = next(reader)
            first_row = next(reader, [])

        print(f"File: {csv_path.name}")
        print(f"Delimiter: {repr(delimiter)}")
        print(f"Columns: {len(header)}")
        print(f"First 8 columns: {header[:8]}")
        print(f"First-row values: {first_row[:8]}")
        print("-" * 72)


if __name__ == "__main__":
    main()