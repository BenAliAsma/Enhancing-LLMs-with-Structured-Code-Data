import os
import json
import ast
from collections import defaultdict
from tree_sitter import Language, Parser
from multilspy import SyncLanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger

# Configuration
project_root = os.path.expanduser("~/Téléchargements/kotlin")
output_file = "project_metadata_multi_kotlin.json"

# Language configuration
language_map = {
    'py': 'python',
    'java': 'java',
    'rs': 'rust',
    'cs': 'csharp',
    'ts': 'typescript',
    'js': 'javascript',
    'go': 'go',
    'dart': 'dart',
    'rb': 'ruby',
    'kt': 'kotlin'
}

# Tree-sitter setup
TREE_SITTER_PARSERS = {}

def load_tree_sitter_parsers():
    try:
        Language.build_library(
            'build/my-languages.so',
            [
                'vendor/tree-sitter-python',
                'vendor/tree-sitter-java',
                'vendor/tree-sitter-rust',
                'vendor/tree-sitter-c-sharp',
                'vendor/tree-sitter-typescript/typescript',
                'vendor/tree-sitter-javascript',
                'vendor/tree-sitter-go',
                'vendor/tree-sitter-dart',
                'vendor/tree-sitter-ruby',
                'vendor/tree-sitter-kotlin',
            ]
        )
        languages = {
            'python': Language('build/my-languages.so', 'python'),
            'java': Language('build/my-languages.so', 'java'),
            'rust': Language('build/my-languages.so', 'rust'),
            'csharp': Language('build/my-languages.so', 'c_sharp'),
            'typescript': Language('build/my-languages.so', 'typescript'),
            'javascript': Language('build/my-languages.so', 'javascript'),
            'go': Language('build/my-languages.so', 'go'),
            'dart': Language('build/my-languages.so', 'dart'),
            'ruby': Language('build/my-languages.so', 'ruby'),
            'kotlin': Language('build/my-languages.so', 'kotlin'),
        }
        for lang, language_obj in languages.items():
            parser = Parser()
            parser.set_language(language_obj)
            TREE_SITTER_PARSERS[lang] = parser
    except Exception as e:
        print(f"Error loading Tree-sitter parsers: {e}")
        raise

load_tree_sitter_parsers()

def extract_ast_metadata(file_path, language):
    """Extract AST metadata using language-specific parsers"""
    metadata = {"functions": [], "classes": [], "imports": []}
    
    # Python AST parsing
    if language == 'python':
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
            print(f"Python AST error in {file_path}: {e}")
        return metadata
    
    # Tree-sitter parsing for other languages
    if language in TREE_SITTER_PARSERS:
        try:
            with open(file_path, "rb") as f:
                code = f.read()
            
            parser = TREE_SITTER_PARSERS[language]
            tree = parser.parse(code)
            
            # Language-specific queries
            query = None
            if language == 'java':
                query = """
                (class_declaration name: (identifier) @class)
                (method_declaration name: (identifier) @method)
                (import_statement (scoped_identifier) @import)
                """
            elif language == 'csharp':
                query = """
                (class_declaration name: (identifier) @class)
                (method_declaration name: (identifier) @method)
                (using_directive (qualified_name) @import)
                """
            elif language == 'kotlin':
                query = """
                (class_declaration (simple_identifier) @class)
                (function_declaration (simple_identifier) @function)
                (import_header (identifier) @import)
                """
            elif language in ['typescript', 'javascript']:
                query = """
                (function_declaration name: (identifier) @function)
                (class_declaration name: (identifier) @class)
                (import_statement source: (string) @import)
                """
            elif language == 'rust':
                query = """
                (function_item name: (identifier) @function)
                (struct_item name: (identifier) @class)
                (use_declaration (scoped_identifier) @import)
                """
            elif language == 'ruby':
                query = """
                ((class name: (constant) @class)
                 (module name: (constant) @class))
                (method name: (identifier) @function)
                (call method: (identifier) @req_method
                    (#match? @req_method "require|require_relative|load")
                    arguments: (argument_list (string (string_content) @import))
                """
            elif language == 'go':
                query = """
                (function_declaration name: (identifier) @function)
                (type_spec name: (type_identifier) @class)
                (import_spec path: (string_lit) @import)
                """
            elif language == 'dart':
                query = """
                (class_declaration name: (identifier) @class)
                (method_declaration name: (identifier) @function)
                (import_specification (uri) @import)
                """
            
            if query:
                lang_query = TREE_SITTER_PARSERS[language].language.query(query)
                captures = lang_query.captures(tree.root_node)
                
                for node, tag in captures:
                    text = node.text.decode()
                    if tag in ['function', 'method']:
                        metadata["functions"].append(text)
                    elif tag == 'class':
                        metadata["classes"].append(text)
                    elif tag == 'import':
                        metadata["imports"].append(text)
                    elif tag == 'req_method':
                        # Handle Ruby requires
                        pass  # Already captured through string_content
                    
        except Exception as e:
            print(f"Tree-sitter error in {file_path} ({language}): {e}")
    
    return metadata

def safe_lsp_request(request_func, *args):
    """Safely execute LSP requests with error handling"""
    try:
        result = request_func(*args)
        return result if result is not None else []
    except Exception as e:
        print(f"LSP error: {e}")
        return []

def process_file(file_path, language, lsp, project_root):
    """Process a single file and extract metadata"""
    relative_path = os.path.relpath(file_path, project_root)
    return {
        "file": relative_path,
        "language": language,
        "symbols": safe_lsp_request(lsp.request_document_symbols, relative_path),
        "references": [],
        "definitions": [],
        "ast": extract_ast_metadata(file_path, language)
    }

def main():
    metadata_collection = {"project_structure": [], "files": []}
    language_files = defaultdict(list)

    # Collect project structure and categorize files by language
    for root, dirs, files in os.walk(project_root):
        rel_root = os.path.relpath(root, project_root)
        metadata_collection["project_structure"].append({
            "directory": rel_root,
            "files": files
        })
        
        for f in files:
            ext = os.path.splitext(f)[1].lstrip('.')
            if lang := language_map.get(ext):
                file_path = os.path.join(root, f)
                language_files[lang].append(file_path)

    # Process files by language
    for lang, files in language_files.items():
        config = MultilspyConfig.from_dict({
            "code_language": lang,
            "lsp_servers": {
                "csharp": {"command": "OmniSharp"},
                "typescript": {"command": "typescript-language-server"},
                "javascript": {"command": "typescript-language-server"},
                "ruby": {"command": "solargraph"},
                "kotlin": {"command": "kotlin-language-server"}
            }
        })
        logger = MultilspyLogger(log_level="debug")
        
        try:
            lsp = SyncLanguageServer.create(config, logger, project_root)
            with lsp.start_server():
                print(f"Processing {len(files)} {lang} files...")
                
                for file_path in files:
                    metadata = process_file(file_path, lang, lsp, project_root)
                    metadata_collection["files"].append(metadata)
        except Exception as e:
            print(f"Error processing {lang} files: {str(e)}")
            logger.log(f"Detailed error: {str(e)}", "error")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata_collection, f, indent=4, ensure_ascii=False)
    
    print(f"Metadata extraction complete! Saved to {output_file}")

if __name__ == "__main__":
    main()