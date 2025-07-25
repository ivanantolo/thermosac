#!/usr/bin/env python3

import re, sys
from pathlib import Path
from typing import List, Dict
import pandas as pd

# ---------------------------------------------------------
LLE_FILE = "LLeDbPGL6ed96b.txt"   # optionally adjust Path/Name
CSV_SEP  = ";"                    # CSV-Separator
# ---------------------------------------------------------

HDR_RE = re.compile(r'^\s*99\s+13\s+(\d+)\s+(\d+)(?:\s+!\s*(.+))?')

def cas_hyphen(raw: str | int) -> str:
    """7732185  -->  7732-18-5"""
    s = str(raw)
    return f"{int(s[:-3])}-{s[-3:-1]}-{s[-1]}"

def split_names(desc: str | None) -> List[str]:
    if not desc:
        return []
    desc = re.sub(r'\b\d{4}\b.*$', '', desc).strip()
    return [p.strip() for p in re.split(r'[+/,;&]+', desc) if p.strip()]

def parse_lle(path: Path) -> pd.DataFrame:
    txt = path.read_text(encoding="utf-8", errors="ignore")

    rows: List[Dict] = []
    cas_name: Dict[str, str] = {}

    for block in txt.split("*** Next ***"):
        block = block.strip()
        if not block:
            continue
        first, *data_lines = block.splitlines()
        m = HDR_RE.match(first.strip())
        if not m:
            continue

        cas1_raw, cas2_raw, desc = m.groups()
        cas1_raw, cas2_raw = cas1_raw.strip(), cas2_raw.strip()
        names = split_names(desc)

        # Mapping CAS --> Name
        if len(names) >= 1:
            cas_name.setdefault(cas1_raw, names[0])
        if len(names) >= 2:
            cas_name.setdefault(cas2_raw, names[1])

        for ln in data_lines:
            ln = ln.strip()
            if not ln or ln.startswith('*') or HDR_RE.match(ln):
                continue
            toks = re.split(r'\s+', ln)
            if len(toks) < 5:
                continue
            try:
                T, P, phase, x1_L1, x1_L2 = map(float, toks[:5])
            except ValueError:
                continue
            source = " ".join(toks[5:]) if len(toks) > 5 else ""
            rows.append(dict(
                cas1=cas_hyphen(cas1_raw),
                name1=cas_name.get(cas1_raw, ""),
                cas2=cas_hyphen(cas2_raw),
                name2=cas_name.get(cas2_raw, ""),
                T_K=T, P_kPa=P, phase=int(phase),
                x1_L1=x1_L1, x1_L2=x1_L2, source=source
            ))

    return pd.DataFrame(rows)

def main() -> None:
    in_path = Path(LLE_FILE)
    if not in_path.exists():
        sys.exit(f"[error] Input file '{LLE_FILE}' not found")

    df = parse_lle(in_path)
    out_csv = in_path.with_suffix(".csv")
    df.to_csv(out_csv, sep=CSV_SEP, index=False)
    print(f"[x] {len(df):,} Rows --> {out_csv}")

if __name__ == "__main__":
    main()
