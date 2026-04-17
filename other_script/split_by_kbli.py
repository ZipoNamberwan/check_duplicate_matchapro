from __future__ import annotations

from pathlib import Path
import time

import pandas as pd


DEFAULT_INPUT = Path("source_matcha_pro_all") / "combined_data.csv"

DEFAULT_OUTPUT_IN = Path("result") / "combined_all_in.csv"
DEFAULT_OUTPUT_OUT = Path("result") / "combined_all_out.csv"

DEFAULT_ALLOWED = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U"]

DEFAULT_KBLI_PREFIXES: list[str] = ['01', '02', '03', '87', '88', '92', '96999', '9412', '942', '949']

DEFAULT_CHUNKSIZE = 250_000
DEFAULT_PROGRESS_EVERY = 1


def iter_csv_batches(path: Path, *, chunksize: int) -> pd.io.parsers.TextFileReader:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    return pd.read_csv(path, dtype=str, chunksize=chunksize, low_memory=False)


def extract_kategori_from_kegiatan_usaha(series: pd.Series) -> pd.Series:

    pattern = r"Kategori\s*:\s*([A-Za-z])\b"

    extracted = series.fillna("").astype(str).str.extract(pattern, expand=False)

    return extracted.astype("string").str.upper()


def extract_kbli_from_kegiatan_usaha(series: pd.Series) -> pd.Series:

    pattern = r"KBLI\s*:\s*([0-9]+)\b"

    extracted = series.fillna("").astype(str).str.extract(pattern, expand=False)

    return extracted.astype("string").str.strip()


def infer_kategori_from_kbli(kbli_series: pd.Series) -> pd.Series:

    prefix2 = kbli_series.fillna("").str[:2]

    num = pd.to_numeric(prefix2, errors="coerce")

    kategori = pd.Series(pd.NA, index=kbli_series.index, dtype="string")

    kategori.loc[num.between(1,4)] = "A"
    kategori.loc[num.between(5,9)] = "B"
    kategori.loc[num.between(10,33)] = "C"
    kategori.loc[num == 35] = "D"
    kategori.loc[num.between(36,39)] = "E"
    kategori.loc[num.between(41,43)] = "F"
    kategori.loc[num.between(45,47)] = "G"
    kategori.loc[num.between(49,53)] = "H"
    kategori.loc[num.between(55,56)] = "I"
    kategori.loc[num.between(58,63)] = "J"
    kategori.loc[num.between(64,66)] = "K"
    kategori.loc[num == 68] = "L"
    kategori.loc[num.between(69,75)] = "M"
    kategori.loc[num.between(77,82)] = "N"
    kategori.loc[num == 84] = "O"
    kategori.loc[num == 85] = "P"
    kategori.loc[num.between(86,88)] = "Q"
    kategori.loc[num.between(90,93)] = "R"
    kategori.loc[num.between(94,96)] = "S"

    return kategori


def process_in_batches(
    input_path: Path,
    output_in_path: Path,
    output_out_path: Path,
    *,
    allowed: list[str],
    kbli_prefixes: list[str],
    chunksize: int,
) -> tuple[int, int, int]:

    t0 = time.perf_counter()

    allowed_set = {str(x).strip().upper() for x in allowed if str(x).strip()}

    kbli_prefix_tuple = tuple(str(x).strip() for x in kbli_prefixes if str(x).strip())

    output_in_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_kept = 0
    total_removed = 0

    wrote_header = False

    print(f"reading={input_path}")
    print(f"writing_in={output_in_path}")
    print(f"writing_out={output_out_path}")
    print(f"chunksize={chunksize}")

    for chunk_idx, chunk in enumerate(iter_csv_batches(input_path, chunksize=chunksize), start=1):

        total_in += len(chunk)

        if "kegiatan_usaha" not in chunk.columns:
            raise KeyError("Column not found: kegiatan_usaha")

        chunk["kategori"] = extract_kategori_from_kegiatan_usaha(chunk["kegiatan_usaha"])

        chunk["kbli"] = extract_kbli_from_kegiatan_usaha(chunk["kegiatan_usaha"])

        kategori_blank = chunk["kategori"].isna() | chunk["kategori"].astype("string").str.strip().isin(["", "-"])

        inferred = infer_kategori_from_kbli(chunk["kbli"])

        chunk.loc[kategori_blank, "kategori"] = inferred[kategori_blank]

        kategori_is_allowed = chunk["kategori"].isin(allowed_set)

        kategori_blank_or_dash = chunk["kategori"].isna() | chunk["kategori"].astype("string").str.strip().isin(["", "-"])

        kbli_matches_prefix = (
            chunk["kbli"].fillna("").astype(str).str.startswith(kbli_prefix_tuple)
            if kbli_prefix_tuple
            else pd.Series(False, index=chunk.index)
        )

        mask_keep = kategori_is_allowed | (kategori_blank_or_dash & kbli_matches_prefix)

        kept = chunk[mask_keep].copy()

        removed = chunk[~mask_keep].copy()

        chunk_kept = len(kept)
        chunk_removed = len(removed)

        total_kept += chunk_kept
        total_removed += chunk_removed

        mode = "w" if not wrote_header else "a"

        if chunk_kept > 0:
            kept.to_csv(output_in_path, index=False, mode=mode, header=not wrote_header)

        if chunk_removed > 0:
            removed.to_csv(output_out_path, index=False, mode=mode, header=not wrote_header)

        wrote_header = True

        if chunk_idx % DEFAULT_PROGRESS_EVERY == 0:

            elapsed = time.perf_counter() - t0

            print(
                f"chunk={chunk_idx} "
                f"read={len(chunk)} "
                f"kept={chunk_kept} "
                f"removed={chunk_removed} "
                f"total_in={total_in} "
                f"total_kept={total_kept} "
                f"total_removed={total_removed} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0

    print(
        f"done total_in={total_in} "
        f"kept={total_kept} "
        f"removed={total_removed} "
        f"elapsed_s={elapsed:.1f}"
    )

    return total_in, total_kept, total_removed


def main() -> None:

    total_in, total_kept, total_removed = process_in_batches(
        DEFAULT_INPUT,
        DEFAULT_OUTPUT_IN,
        DEFAULT_OUTPUT_OUT,
        allowed=DEFAULT_ALLOWED,
        kbli_prefixes=DEFAULT_KBLI_PREFIXES,
        chunksize=DEFAULT_CHUNKSIZE,
    )

    print(f"input_rows={total_in}")
    print(f"kept_rows={total_kept}")
    print(f"removed_rows={total_removed}")
    print(f"saved_kept={DEFAULT_OUTPUT_IN}")
    print(f"saved_removed={DEFAULT_OUTPUT_OUT}")


if __name__ == "__main__":
    main()