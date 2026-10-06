"""Read text from .txt / .md / .pdf / .docx files (no heavy dependencies)."""
import re
import zipfile
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        if ext == ".docx":
            with zipfile.ZipFile(path) as z:
                xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
            xml = re.sub(r"</w:p>", "\n", xml)
            return re.sub(r"<[^>]+>", "", xml)
    except Exception as e:  # corrupted / encrypted files shouldn't kill indexing
        print(f"[skip] {path.name}: {e}")
    return ""


def scan_folder(folder: str):
    """Yield (path, mtime) for every supported file under folder."""
    for p in sorted(Path(folder).expanduser().rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("."):
            yield p, p.stat().st_mtime
