import os
from langchain.tools import tool

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@tool
def read_file_content(path: str) -> str:
    """Read the contents of a file by its path."""
    path = path.strip().strip('"\'').split("(")[0].split("\n")[0].strip()

    if not os.path.isabs(path):
        path = os.path.join(WORK_DIR, path)

    if not os.path.exists(path):
        return f"File not found: {path}"

    size = os.path.getsize(path)
    if size > 500_000:
        return f"File is too large ({size // 1024} KB). Max allowed: 500 KB."

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return "Cannot read file: unsupported binary format."
    except Exception as e:
        return f"Error reading file: {e}"
