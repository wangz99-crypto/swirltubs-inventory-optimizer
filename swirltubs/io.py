import pandas as pd

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

import pandas as pd

REQ_FULL = ["Part num", "Part Size (Ft^3)", "Part cost", "Avg Annual Use"]

def _find_header_row(df_raw: pd.DataFrame, required: list[str]) -> int:
    """
    df_raw: sheet read with header=None
    Return the row index that contains the header names.
    """
    required_set = {c.strip() for c in required}
    for r in range(min(len(df_raw), 80)):  # scan first 80 rows is enough
        row_vals = [
            str(x).strip() for x in df_raw.iloc[r].tolist()
            if pd.notna(x) and str(x).strip() != ""
        ]
        row_set = set(row_vals)
        # if this row contains ALL required headers, it's the header row
        if required_set.issubset(row_set):
            return r
    return -1

def load_part1(path: str, mode: str = "full") -> pd.DataFrame:
    """
    mode:
      - "full": sheet 'Raw Data' with header row at row 1
      - "test": sheet 'Test Data' (30 parts)
    Output columns: part, size, cost, annual_use
    """
    if mode == "test":
        df = pd.read_excel(path, sheet_name="Test Data")
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns={
            "Part num": "part",
            "Part Size (Ft^3)": "size",
            "Part cost": "cost",
            "Annual Use": "annual_use",
        })
        return df[["part", "size", "cost", "annual_use"]].copy()

    if mode == "full":
        # 1) read without header so we can detect where header is
        df_raw = pd.read_excel(path, sheet_name="Raw Data", header=None)

        header_row = _find_header_row(df_raw, REQ_FULL)
        if header_row < 0:
            raise ValueError(
                f"Could not find header row in 'Raw Data'. "
                f"Expected columns {REQ_FULL}. "
                f"Tip: check spelling/case in Excel."
            )

        # 2) read again using the detected header row
        df = pd.read_excel(path, sheet_name="Raw Data", header=header_row)
        df.columns = [str(c).strip() for c in df.columns]

        missing = [c for c in REQ_FULL if c not in df.columns]
        if missing:
            raise ValueError(f"Raw Data missing columns: {missing}. Found: {list(df.columns)}")

        out = df[REQ_FULL].rename(columns={
            "Part num": "part",
            "Part Size (Ft^3)": "size",
            "Part cost": "cost",
            "Avg Annual Use": "annual_use",
        }).copy()

        # numeric cleanup
        for c in ["part", "size", "cost", "annual_use"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["part", "size", "cost", "annual_use"]).reset_index(drop=True)
        out["part"] = out["part"].astype(int)

        return out

    raise ValueError("mode must be 'full' or 'test'")