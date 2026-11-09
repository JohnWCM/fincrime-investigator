"""Safe filename literal for repository-managed local paths."""
from pathlib import Path

def sql_path(path: Path) -> str:
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"
