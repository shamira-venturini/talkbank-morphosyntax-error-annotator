import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Union


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Union[str, Path], records: Iterable[Dict], mode: str = "w") -> int:
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
