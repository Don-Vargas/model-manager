import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class RepositoryDiscoverer:
    """Responsable únicamente de escanear el sistema de archivos y cargar metadatos clave."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.model_info = self._load_json(repo_dir / "model_info.json")
        self.model_index = self._load_json(repo_dir / "model_index.json")

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            resolved = path.resolve()
            if resolved.exists():
                with open(resolved, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def get_all_files(self) -> List[str]:
        if self.model_info and "siblings" in self.model_info:
            return [s["rfilename"] for s in self.model_info["siblings"]]
        return [
            str(p.relative_to(self.repo_dir)).replace("\\", "/")
            for p in self.repo_dir.rglob("*")
            if p.is_file()
        ]

    def get_file_sizes(self) -> Dict[str, Optional[int]]:
        if self.model_info and "siblings" in self.model_info:
            return {
                s["rfilename"]: s.get("size")
                for s in self.model_info["siblings"]
            }
        return {}
