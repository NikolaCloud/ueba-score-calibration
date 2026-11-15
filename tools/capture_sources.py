"""Save accessible primary sources and explicit access failures for the audit trail."""
import concurrent.futures
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "facade.pdf": "https://www.usenix.org/system/files/usenixsecurity26-kantchelian.pdf",
    "martinez_gil.pdf": "https://raw.githubusercontent.com/mlresearch/v244/main/assets/martinez-gil24a/martinez-gil24a.pdf",
    "bowie_university.html": "https://researchonline.gcu.ac.uk/en/publications/counterfactual-routine-deviation-personalised-behavioural-modelli/",
    "bowie_full.pdf": "https://researchonline.gcu.ac.uk/files/116676993/116650450.pdf",
    "session_paper.html": "https://www.sciencedirect.com/science/article/pii/S2666827026001118",
    "hcc_cp.html": "https://ww2.amstat.org/meetings/jsm/2026/onlineprogram/abstract.cfm?sid=8751&tid=9087",
    "weighted_low_data.html": "https://arxiv.org/html/2603.23205v1",
}


def main():
    folder = ROOT / "data/literature"
    folder.mkdir(parents=True, exist_ok=True)
    def fetch(item):
        name, url = item
        try:
            response = requests.get(url, timeout=60)
            entry = dict(name=name, url=url, status=response.status_code, bytes=len(response.content))
            if response.status_code == 200:
                payload = response.content
                (folder / name).write_bytes(payload)
                entry["sha256"] = hashlib.sha256(payload).hexdigest()
                if name.endswith(".pdf"):
                    import pymupdf
                    doc = pymupdf.open(stream=payload, filetype="pdf")
                    (folder / (name+".txt")).write_text("\n".join(page.get_text() for page in doc), encoding="utf-8")
            return entry
        except Exception as exc:
            return dict(name=name, url=url, error=str(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, SOURCES.items()))
    (ROOT / "data/metadata/literature_access.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for entry in results:
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
