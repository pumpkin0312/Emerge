"""
Download protein sequences from UniProt REST API by taxonomy ID.

Usage:
    python data/download_data.py              # 使用config/config.yaml默认设置
    python data/download_data.py --max 5000   # 每个物种限制5000条
    python data/download_data.py --taxa 2 4751 9606  # 指定taxa ID
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"


# -----------------------------------------------------------------------
def download_taxon(
    taxon_id: int,
    taxon_name: str,
    out_dir: Path,
    max_seqs: int | None = None,
    min_len: int = 50,
    max_len: int = 512,
    exclude_taxon_id: int | None = None,
) -> Path:
    out_path = out_dir / f"{taxon_name.lower()}_{taxon_id}.fasta"
    if out_path.exists():
        logger.info(f"  Already exists: {out_path}, skipping download")
        return out_path

    query = f"taxonomy_id:{taxon_id} AND reviewed:true AND length:[{min_len} TO {max_len}]"
    if exclude_taxon_id is not None:
        query += f" AND NOT taxonomy_id:{exclude_taxon_id}"
    size  = min(500, max_seqs) if max_seqs else 500

    params = {
        "query":  query,
        "format": "fasta",
        "size":   size,
    }

    url     = f"{UNIPROT_API}?{urlencode(params)}"
    fetched = 0
    headers = {"User-Agent": "emerge-protein-lm/1.0 (research project)"}

    logger.info(f"Downloading {taxon_name} (taxon {taxon_id}) → {out_path}")

    with open(out_path, "w", encoding="utf-8") as fout:
        while url:
            req  = Request(url, headers=headers)
            try:
                resp = urlopen(req, timeout=60)
            except URLError as e:
                logger.error(f"  Network error: {e}. Retrying in 10s…")
                time.sleep(10)
                resp = urlopen(req, timeout=60)

            data = resp.read().decode("utf-8")

            # parse number of sequences in this page
            page_seqs = data.count(">")
            fout.write(data)
            fetched += page_seqs
            logger.info(f"  {taxon_name}: {fetched} sequences downloaded…")

            if max_seqs and fetched >= max_seqs:
                break

            # follow UniProt pagination via Link header
            link_header = resp.headers.get("Link", "")
            next_url    = _parse_next_link(link_header)
            url         = next_url
            if url:
                time.sleep(0.3)  # polite delay

    logger.info(f"  Done: {fetched} sequences saved to {out_path}")
    return out_path


def _parse_next_link(link_header: str) -> str | None:
    """Parse 'Link: <url>; rel="next"' header."""
    if not link_header:
        return None
    parts = link_header.split(",")
    for part in parts:
        if 'rel="next"' in part:
            start = part.find("<") + 1
            end   = part.find(">")
            if start > 0 and end > start:
                return part[start:end].strip()
    return None


# -----------------------------------------------------------------------
def filter_fasta(
    in_path: Path,
    out_path: Path,
    min_len: int,
    max_len: int,
    max_seqs: int | None,
    valid_aa: str = "ACDEFGHIKLMNPQRSTVWY",
) -> int:
    """Filter sequences by length and valid amino acids."""
    valid_set = set(valid_aa)
    kept = 0
    current_header, current_seq = "", []

    def _write_if_valid(fout, header, seq):
        nonlocal kept
        s = "".join(seq).upper()
        if not (min_len <= len(s) <= max_len):
            return
        if not all(c in valid_set for c in s):
            return
        if max_seqs and kept >= max_seqs:
            return
        fout.write(f"{header}\n{s}\n")
        kept += 1

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.rstrip()
            if line.startswith(">"):
                if current_header:
                    _write_if_valid(fout, current_header, current_seq)
                current_header = line
                current_seq    = []
            else:
                current_seq.append(line)
        if current_header:
            _write_if_valid(fout, current_header, current_seq)

    return kept


# -----------------------------------------------------------------------
def main():
    cfg  = load_config()
    dcfg = cfg["data"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--max",   type=int,   default=None)
    parser.add_argument("--taxa",  type=int,   nargs="+", default=None)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    raw_dir  = Path(dcfg["raw_dir"])
    proc_dir = Path(dcfg["processed_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    max_seqs = args.max or dcfg.get("max_seqs_per_taxon")
    min_len  = dcfg["min_seq_len"]
    max_len  = dcfg["max_seq_len"]

    # Combine training and evaluation taxa
    all_taxa = dcfg["train_taxa"] + dcfg["eval_taxa"]

    if args.taxa:
        all_taxa = [t for t in all_taxa if t["id"] in args.taxa]

    for taxon in all_taxa:
        raw_path = download_taxon(
            taxon_id   = taxon["id"],
            taxon_name = taxon["name"],
            out_dir    = raw_dir,
            max_seqs   = max_seqs,
            min_len    = min_len,
            max_len    = max_len,
        )
        # filter pass
        proc_path = proc_dir / raw_path.name
        kept = filter_fasta(raw_path, proc_path, min_len, max_len, max_seqs)
        logger.info(f"  Filtered: {kept} sequences → {proc_path}")

    logger.info("All downloads complete.")


if __name__ == "__main__":
    main()
