import os
import json
import ast
from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger

# Chemin du projet
project_root = os.path.expanduser("~/Téléchargements/astropy-main")

# Configuration de multilspy pour Python
config = MultilspyConfig.from_dict({"code_language": "python"})
logger = MultilspyLogger()

# Création du client LSP
lsp = SyncLanguageServer.create(config, logger, project_root)

def extract_ast_metadata(file_path):
    """Extrait les métadonnées AST pour une meilleure compréhension du projet."""
    metadata = {"functions": [], "classes": [], "imports": []}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metadata["functions"].append(node.name)
            elif isinstance(node, ast.ClassDef):
                metadata["classes"].append(node.name)
            elif isinstance(node, ast.Import):
                metadata["imports"].extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                metadata["imports"].extend(f"{module}.{alias.name}" for alias in node.names)
    except Exception as e:
        print(f"Erreur AST dans {file_path}: {e}")
    
    return metadata

def safe_lsp_request(request_func, *args):
    """Exécute une requête LSP en toute sécurité en gérant les erreurs et les valeurs None."""
    try:
        result = request_func(*args)
        return result if result is not None else []
    except Exception as e:
        print(f"Erreur LSP: {e}")
        return []

def extract_metadata(file_path):
    """Extrait toutes les métadonnées d'un fichier Python pour une meilleure compréhension par un LLM."""
    relative_path = os.path.relpath(file_path, project_root)
    metadata = {
        "file": relative_path,
        "symbols": [],
        "references": [],
        "definitions": [],
        "ast": extract_ast_metadata(file_path)
    }
    
    try:
        # Extraction des symboles du fichier (fonctions, classes, variables globales)
        metadata["symbols"] = safe_lsp_request(lsp.request_document_symbols, relative_path)
        
        # Extraction des définitions (où les éléments sont définis dans le projet)
        metadata["definitions"] = [
            safe_lsp_request(lsp.request_definition, relative_path, sym["range"]["start"]["line"], sym["range"]["start"]["character"])
            for sym in metadata["symbols"] if "range" in sym
        ]
        
        # Extraction des références (où ces éléments sont utilisés)
        metadata["references"] = [
            safe_lsp_request(lsp.request_references, relative_path, sym["range"]["start"]["line"], sym["range"]["start"]["character"], True)
            for sym in metadata["symbols"] if "range" in sym
        ]
        
    except Exception as e:
        print(f"Erreur LSP dans {relative_path}: {e}")
    
    return metadata

if __name__ == "__main__":
    metadata_collection = {
        "project_structure": [],
        "files": []
    }
    
    # Démarrage du serveur LSP
    with lsp.start_server():
        # Parcours des fichiers du projet
        for root, dirs, files in os.walk(project_root):
            rel_root = os.path.relpath(root, project_root)
            metadata_collection["project_structure"].append({"directory": rel_root, "files": files})
            for f in files:
                if f.endswith('.py'):
                    file_path = os.path.join(root, f)
                    metadata_collection["files"].append(extract_metadata(file_path))
    
    # Sauvegarde des métadonnées dans un fichier JSON
    with open("project_metadata_astropy.json", "w", encoding="utf-8") as f:
        json.dump(metadata_collection, f, indent=4)
    
    print("Extraction terminée ! Les métadonnées sont enregistrées dans project_metadata.json")
