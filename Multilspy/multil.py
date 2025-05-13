import os
import json
import ast
import logging
from pathlib import Path
from typing import Dict, Any
from multilspy.multilspy_logger import MultilspyLogger


from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig

from multilspy.multilspy_logger import MultilspyLogger

class MetadataExtractor:
    """Extracteur de métadonnées amélioré avec gestion d'erreurs renforcée"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).expanduser()
        self.config = MultilspyConfig.from_dict({"code_language": "python"})
        self.logger = self._setup_logger()
        self.lsp = SyncLanguageServer.create(self.config, self.logger, str(self.project_root))

    def _setup_logger(self) -> MultilspyLogger:
        """Configure le logger avec des niveaux appropriés"""
        logger = MultilspyLogger()
        # Assuming MultilspyLogger uses Python's standard logging setup
        logger.logger.setLevel(logging.INFO)  # Use setLevel() for standard logging
        return logger



    def _safe_lsp_request(self, request_func, *args) -> list:
        """Exécute une requête LSP avec gestion robuste des erreurs"""
        try:
            result = request_func(*args)
            return result if isinstance(result, list) else []
        except Exception as e:
            self.logger.error(f"Erreur LSP: {str(e)}")
            return []

    def extract_ast_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extrait les métadonnées AST avec gestion des erreurs de syntaxe"""
        metadata = {
            "functions": [],
            "classes": [],
            "imports": [],
            "errors": []
        }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    try:
                        if isinstance(node, ast.FunctionDef):
                            metadata["functions"].append({
                                "name": node.name,
                                "lineno": node.lineno,
                                "args": [a.arg for a in node.args.args],
                                "returns": ast.unparse(node.returns) if node.returns else None
                            })
                        elif isinstance(node, ast.ClassDef):
                            metadata["classes"].append({
                                "name": node.name,
                                "bases": [ast.unparse(b) for b in node.bases],
                                "lineno": node.lineno
                            })
                        elif isinstance(node, (ast.Import, ast.ImportFrom)):
                            imports = []
                            if isinstance(node, ast.Import):
                                imports.extend(n.name for n in node.names)
                            else:
                                module = node.module or ""
                                imports.extend(f"{module}.{n.name}" for n in node.names)
                            metadata["imports"].extend(imports)
                    except Exception as e:
                        metadata["errors"].append({
                            "type": "AST Processing",
                            "message": str(e),
                            "lineno": getattr(node, "lineno", None)
                        })

        except SyntaxError as e:
            metadata["errors"].append({
                "type": "Syntax Error",
                "message": f"{e.msg} (line {e.lineno})",
                "lineno": e.lineno
            })
        except Exception as e:
            metadata["errors"].append({
                "type": "General Error",
                "message": str(e)
            })

        return metadata

    def extract_lsp_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extrait les métadonnées LSP avec validation des données"""
        relative_path = str(file_path.relative_to(self.project_root))
        metadata = {
            "file": relative_path,
            "symbols": [],
            "references": [],
            "definitions": [],
            "errors": []
        }

        try:
            # Extraction des symboles
            symbols = self._safe_lsp_request(
                self.lsp.request_document_symbols,
                relative_path
            )

            for sym in symbols:
                try:
                    if not isinstance(sym, dict) or "range" not in sym:
                        continue

                    # Extraction des définitions
                    defs = self._safe_lsp_request(
                        self.lsp.request_definition,
                        relative_path,
                        sym["range"]["start"]["line"],
                        sym["range"]["start"]["character"]
                    )
                    metadata["definitions"].extend(defs)

                    # Extraction des références
                    refs = self._safe_lsp_request(
                        self.lsp.request_references,
                        relative_path,
                        sym["range"]["start"]["line"],
                        sym["range"]["start"]["character"],
                        True
                    )
                    metadata["references"].extend(refs)

                except Exception as e:
                    metadata["errors"].append({
                        "type": "Symbol Processing",
                        "message": str(e),
                        "symbol": str(sym)[:100]
                    })

            metadata["symbols"] = symbols

        except Exception as e:
            metadata["errors"].append({
                "type": "LSP Error",
                "message": str(e)
            })

        return metadata

    def generate_metadata(self) -> Dict[str, Any]:
        metadata = {
        "project_structure": [],
        "files": [],
        "stats": {
            "total_files": 0,
            "error_files": 0
        }
    }

        with self.lsp.start_server():
            for root, dirs, files in os.walk(str(self.project_root)):
                rel_root = Path(root).relative_to(self.project_root)
                entry = {
                "directory": str(rel_root),
                "files": [],
                "subdirectories": dirs
            }

                for f in files:
                    file_path = Path(root) / f
                    if file_path.suffix == ".py":
                        entry["files"].append(f)
                        file_meta = {
                        "path": str(file_path.relative_to(self.project_root)),
                        "ast": self.extract_ast_metadata(file_path),
                        "lsp": self.extract_lsp_metadata(file_path)
                    }
                    
                    # Ensure that file_meta['ast'] and file_meta['lsp'] are dictionaries and contain 'errors' key
                        if isinstance(file_meta['ast'], dict) and 'errors' not in file_meta['ast']:
                            file_meta['ast']["errors"] = []
                        if isinstance(file_meta['lsp'], dict) and 'errors' not in file_meta['lsp']:
                            file_meta['lsp']["errors"] = []

                    # Ensure that file_meta is a dictionary with 'ast' and 'lsp' being dictionaries
                        if isinstance(file_meta, dict):
                            metadata["files"].append(file_meta)
                            metadata["stats"]["total_files"] += 1

                        # Check if any of the 'ast' or 'lsp' has errors
                            if any(len(x.get("errors", [])) > 0 for x in file_meta.values() if isinstance(x, dict)):
                                metadata["stats"]["error_files"] += 1

                    metadata["project_structure"].append(entry)

        return metadata


    def save_metadata(self, output_path: str = "project_metadata_multil.json"):
        """Sauvegarde les métadonnées dans un fichier JSON"""
        metadata = self.generate_metadata()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    PROJECT_ROOT = "~/LSP/local-deep-researcher-main/src/ollama_deep_researcher"
    
    extractor = MetadataExtractor(PROJECT_ROOT)
    extractor.save_metadata()
    
    print("Extraction des métadonnées terminée avec succès!")
    print(f"Résultats sauvegardés dans project_metadata_multil.json")
