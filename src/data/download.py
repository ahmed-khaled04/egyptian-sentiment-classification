"""Download the raw public datasets into data/raw/ (skips files that already exist).

    python -m src.data.download
"""

import urllib.request
from pathlib import Path

RAW_DIR = Path("data/raw")

SOURCES = {
    "astd/Tweets.txt": "https://raw.githubusercontent.com/mahmoudnabil/ASTD/master/data/Tweets.txt",
    "arsarcasm/training_data.csv": "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/training_data.csv",
    "arsarcasm/testing_data.csv": "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/testing_data.csv",
}


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    for rel, url in SOURCES.items():
        dest = raw_dir / rel
        if dest.exists() and not force:
            print(f"skip  {dest} (exists)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print(f"saved {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    download()
