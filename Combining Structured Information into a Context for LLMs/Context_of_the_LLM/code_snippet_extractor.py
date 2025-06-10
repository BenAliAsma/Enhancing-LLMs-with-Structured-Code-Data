"""
Code Snippet Extractor Module

This module extracts code snippets from matched blocks and generates markdown files
with organized code snippets for each entity.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from src.config import commit, date, version, repo_name, problem_stmt


class CodeSnippetExtractor:
    """Extract and organize code snippets from matched blocks."""
    
    def __init__(self, repo_root: str = None, output_dir: str = "outputs"):
        """
        Initialize the CodeSnippetExtractor.
        
        Args:
            repo_root (str): Root directory of the repository (defaults to repo_name from config)
            output_dir (str): Directory to save output files (default: "outputs")
        """
        self.repo_root = repo_root or repo_name.split('/')[-1]  # Extract repo name from "owner/repo"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"CodeSnippetExtractor initialized:")
        print(f"  Repository root: {self.repo_root}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Working with commit: {commit}")
    
    def extract_path_from_symbol(self, symbol: str) -> Optional[str]:
        """
        Extract file path from symbol string.
        
        Args:
            symbol (str): Symbol string containing module path
            
        Returns:
            str: File path or None if extraction fails
        """
        try:
            # Extract module part between backticks
            module_part = symbol.split('`')[1]
            path = module_part.replace('.', '/') + '.py'
            
            # Fix duplicate repository name in path
            repo_name_only = self.repo_root
            duplicate_prefix = f"{repo_name_only}/{repo_name_only}/"
            if path.startswith(duplicate_prefix):
                path = path.replace(duplicate_prefix, f"{repo_name_only}/", 1)
                
            return path
        except Exception as e:
            print(f"⚠ Symbol path parsing error: {symbol} -> {e}")
            return None
    
    def extract_snippet(self, file_path: str, enclosing_range: List[int]) -> Optional[str]:
        """
        Extract code snippet from file using line/character range.
        
        Args:
            file_path (str): Relative path to the file
            enclosing_range (List[int]): [start_line, start_char, end_line, end_char]
            
        Returns:
            str: Extracted code snippet or None if extraction fails
        """
        if not enclosing_range or len(enclosing_range) != 4:
            print(f"⚠ Invalid enclosing range: {enclosing_range}")
            return None
        
        abs_path = Path(self.repo_root) / file_path
        if not abs_path.is_file():
            print(f"⚠ File not found: {abs_path}")
            return None
        
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            try:
                with open(abs_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"⚠ Error reading file {abs_path}: {e}")
                return None
        
        start_line, start_char, end_line, end_char = enclosing_range
        
        # Ensure line indices are within bounds
        start_line = min(start_line, len(lines) - 1)
        end_line = min(end_line, len(lines) - 1)
        
        # Extract snippet based on range
        if start_line == end_line:
            snippet = lines[start_line][start_char:end_char]
        else:
            snippet_lines = [lines[start_line][start_char:]]
            snippet_lines += lines[start_line + 1:end_line]
            snippet_lines.append(lines[end_line][:end_char])
            snippet = ''.join(snippet_lines)
        
        return snippet.strip('\n')
    
    def process_entity_blocks(self, entity: str, blocks: List[Dict[str, Any]]) -> Tuple[List[Tuple], int]:
        """
        Process all blocks for a given entity and extract unique snippets.
        
        Args:
            entity (str): Entity name
            blocks (List[Dict]): List of block dictionaries
            
        Returns:
            Tuple[List[Tuple], int]: (snippets, skipped_count)
        """
        print(f"\n🔍 Processing entity: {entity} ({len(blocks)} blocks)")
        
        seen_snippets = set()
        snippets = []
        skipped = 0
        
        for block in blocks:
            symbol = block.get("symbol", "")
            enclosing_range = block.get("enclosing_range", None)
            
            if not symbol or not enclosing_range:
                skipped += 1
                continue
            
            path = self.extract_path_from_symbol(symbol)
            if not path:
                skipped += 1
                continue
            
            # Create unique key to avoid duplicates
            snippet_key = (path, tuple(enclosing_range))
            if snippet_key in seen_snippets:
                continue
            
            snippet = self.extract_snippet(path, enclosing_range)
            if snippet:
                snippets.append((path, enclosing_range, snippet))
                seen_snippets.add(snippet_key)
            else:
                skipped += 1
        
        # Sort snippets by file path, then by line number
        snippets.sort(key=lambda x: (x[0], x[1][0]))
        
        return snippets, skipped
    
    def save_entity_snippets(self, entity: str, snippets: List[Tuple]) -> Dict[str, str]:
        """
        Save snippets for an entity to both markdown and Python files.
        
        Args:
            entity (str): Entity name
            snippets (List[Tuple]): List of (path, range, snippet) tuples
            
        Returns:
            Dict[str, str]: Dictionary with 'markdown' and 'python' file paths
        """
        entity_safe = entity.replace('.', '_').replace('-', '_')
        md_file = self.output_dir / f"{entity_safe}.md"
        py_file = self.output_dir / f"{entity_safe}_snippets.py"
        
        # Generate markdown file
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Code Snippets for Entity: {entity}\n\n")
            f.write(f"**Project:** {repo_name}\n")
            f.write(f"**Commit:** {commit}\n")
            f.write(f"**Version:** {version}\n")
            f.write(f"**Date:** {date}\n\n")
            
            if problem_stmt:
                f.write(f"**Problem Statement:**\n{problem_stmt}\n\n")
            
            f.write("---\n\n")
            
            for i, (path, rng, snippet) in enumerate(snippets):
                f.write(f"## Snippet {i+1} — `{path}` — Range: {rng}\n\n")
                f.write(f"**File:** `{path}`\n")
                f.write(f"**Range:** Lines {rng[0]}-{rng[2]}, Chars {rng[1]}-{rng[3]}\n\n")
                f.write("```python\n")
                f.write(snippet)
                f.write("\n```\n\n")
        
        # Generate Python file with executable snippets
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write(f"Code Snippets for Entity: {entity}\n")
            f.write(f"Generated from project: {repo_name}\n")
            f.write(f"Commit: {commit}\n")
            f.write(f"Version: {version}\n")
            f.write(f"Date: {date}\n")
            if problem_stmt:
                f.write(f"\nProblem Statement:\n{problem_stmt}\n")
            f.write('"""\n\n')
            
            # Add imports that might be commonly needed
            f.write("# Common imports (uncomment as needed)\n")
            f.write("# import numpy as np\n")
            f.write("# import matplotlib.pyplot as plt\n")
            f.write("# from astropy.modeling import models as m\n")
            f.write("# from astropy.modeling.separable import separability_matrix\n\n")
            
            # Write each snippet as a separate function or section
            for i, (path, rng, snippet) in enumerate(snippets):
                f.write(f"# " + "="*70 + "\n")
                f.write(f"# SNIPPET {i+1}: {path} (Lines {rng[0]}-{rng[2]})\n")
                f.write(f"# " + "="*70 + "\n\n")
                
                # Try to create a function wrapper if the snippet looks like it could be a function
                snippet_lines = snippet.split('\n')
                if any(line.strip().startswith('def ') for line in snippet_lines):
                    # It's already a function, just add it directly
                    f.write(f"{snippet}\n\n")
                elif any(line.strip().startswith('class ') for line in snippet_lines):
                    # It's a class, add it directly
                    f.write(f"{snippet}\n\n")
                else:
                    # Wrap in a function for easier execution
                    function_name = f"snippet_{i+1}_{path.replace('/', '_').replace('.py', '')}"
                    f.write(f"def {function_name}():\n")
                    f.write(f'    """\n    Snippet from {path} (Lines {rng[0]}-{rng[2]})\n    """\n')
                    
                    # Indent the snippet
                    for line in snippet_lines:
                        if line.strip():  # Don't indent empty lines
                            f.write(f"    {line}\n")
                        else:
                            f.write("\n")
                    f.write("\n")
            
            # Add a main section to demonstrate usage
            f.write("\n" + "="*70 + "\n")
            f.write("# USAGE EXAMPLES\n")
            f.write("="*70 + "\n\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    print(f'Code snippets for entity: {entity}')\n")
            f.write("    print(f'Total snippets: {len(snippets)}')\n")
            f.write("    \n")
            f.write("    # Uncomment to run specific snippets:\n")
            
            for i, (path, rng, snippet) in enumerate(snippets):
                if not any(line.strip().startswith(('def ', 'class ')) for line in snippet.split('\n')):
                    function_name = f"snippet_{i+1}_{path.replace('/', '_').replace('.py', '')}"
                    f.write(f"    # {function_name}()\n")
        
        return {
            'markdown': str(md_file),
            'python': str(py_file)
        }
    
    def create_consolidated_files(self, matched_blocks: Dict[str, List[Dict]], processed_entities: Dict[str, Dict]) -> Dict[str, str]:
        """
        Create consolidated files containing all snippets ranked by entity importance.
        
        Args:
            matched_blocks (Dict): Original matched blocks data
            processed_entities (Dict): Successfully processed entities with file paths
            
        Returns:
            Dict[str, str]: Paths to consolidated markdown and python files
        """
        # Sort entities by number of blocks (descending) - entities with more blocks are more important
        entity_rankings = sorted(
            matched_blocks.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Filter to only include successfully processed entities
        ranked_entities = [(entity, blocks) for entity, blocks in entity_rankings 
                          if entity in processed_entities]
        
        consolidated_md = self.output_dir / "all_snippets_consolidated.md"
        consolidated_py = self.output_dir / "all_snippets_consolidated.py"
        
        # Create consolidated markdown file
        with open(consolidated_md, 'w', encoding='utf-8') as f:
            f.write("# All Code Snippets - Consolidated Report\n\n")
            f.write(f"**Project:** {repo_name}\n")
            f.write(f"**Commit:** {commit}\n")
            f.write(f"**Version:** {version}\n")
            f.write(f"**Date:** {date}\n\n")
            
            if problem_stmt:
                f.write(f"**Problem Statement:**\n{problem_stmt}\n\n")
            
            f.write("---\n\n")
            f.write("## Entity Rankings\n\n")
            f.write("Entities are ranked by number of code blocks (relevance indicator):\n\n")
            
            for rank, (entity, blocks) in enumerate(ranked_entities, 1):
                f.write(f"{rank}. **{entity}** ({len(blocks)} blocks)\n")
            
            f.write("\n---\n\n")
            
            # Write snippets for each entity
            total_snippets = 0
            for rank, (entity, blocks) in enumerate(ranked_entities, 1):
                f.write(f"# Rank {rank}: {entity}\n\n")
                f.write(f"**Relevance Score:** {len(blocks)} blocks\n\n")
                
                # Get snippets for this entity
                snippets, _ = self.process_entity_blocks(entity, blocks)
                total_snippets += len(snippets)
                
                for i, (path, rng, snippet) in enumerate(snippets):
                    f.write(f"## {entity} - Snippet {i+1}\n\n")
                    f.write(f"**File:** `{path}`\n")
                    f.write(f"**Range:** Lines {rng[0]}-{rng[2]}, Chars {rng[1]}-{rng[3]}\n\n")
                    f.write("```python\n")
                    f.write(snippet)
                    f.write("\n```\n\n")
                
                f.write("---\n\n")
            
            f.write(f"\n**Summary:** {len(ranked_entities)} entities, {total_snippets} total snippets\n")
        
        # Create consolidated Python file
        with open(consolidated_py, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write("All Code Snippets - Consolidated Python File\n")
            f.write(f"Generated from project: {repo_name}\n")
            f.write(f"Commit: {commit}\n")
            f.write(f"Version: {version}\n")
            f.write(f"Date: {date}\n")
            if problem_stmt:
                f.write(f"\nProblem Statement:\n{problem_stmt}\n")
            f.write(f"\nEntities ranked by relevance (number of blocks):\n")
            for rank, (entity, blocks) in enumerate(ranked_entities, 1):
                f.write(f"{rank}. {entity} ({len(blocks)} blocks)\n")
            f.write('"""\n\n')
            
            # Add comprehensive imports
            f.write("# Comprehensive imports for all snippets\n")
            f.write("import sys\n")
            f.write("import os\n")
            f.write("from pathlib import Path\n")
            f.write("from typing import Any, Dict, List, Optional, Union, Tuple\n\n")
            f.write("# Domain-specific imports (uncomment as needed)\n")
            f.write("# import numpy as np\n")
            f.write("# import matplotlib.pyplot as plt\n")
            f.write("# from astropy.modeling import models as m\n")
            f.write("# from astropy.modeling.separable import separability_matrix\n")
            f.write("# from astropy import units as u\n")
            f.write("# from astropy.coordinates import SkyCoord\n\n")
            
            # Entity information
            f.write("# " + "="*80 + "\n")
            f.write("# ENTITY RANKINGS AND METADATA\n")
            f.write("# " + "="*80 + "\n\n")
            f.write("ENTITY_RANKINGS = [\n")
            for rank, (entity, blocks) in enumerate(ranked_entities, 1):
                f.write(f"    ({rank}, '{entity}', {len(blocks)}),  # rank, entity, block_count\n")
            f.write("]\n\n")
            
            f.write("def get_entity_info():\n")
            f.write('    """Get information about all entities and their rankings."""\n')
            f.write("    return {\n")
            f.write(f"        'total_entities': {len(ranked_entities)},\n")
            f.write(f"        'project': '{repo_name}',\n")
            f.write(f"        'commit': '{commit}',\n")
            f.write("        'rankings': ENTITY_RANKINGS\n")
            f.write("    }\n\n")
            
            # Write snippets organized by entity
            snippet_counter = 0
            entity_functions = []
            
            for rank, (entity, blocks) in enumerate(ranked_entities, 1):
                entity_safe = entity.replace('.', '_').replace('-', '_')
                f.write("# " + "="*80 + "\n")
                f.write(f"# RANK {rank}: {entity} ({len(blocks)} blocks)\n")
                f.write("# " + "="*80 + "\n\n")
                
                snippets, _ = self.process_entity_blocks(entity, blocks)
                entity_snippet_functions = []
                
                for i, (path, rng, snippet) in enumerate(snippets):
                    snippet_counter += 1
                    function_name = f"rank_{rank:02d}_{entity_safe}_snippet_{i+1}"
                    entity_snippet_functions.append(function_name)
                    
                    f.write(f"def {function_name}():\n")
                    f.write(f'    """\n')
                    f.write(f'    Rank {rank} | Entity: {entity}\n')
                    f.write(f'    File: {path} (Lines {rng[0]}-{rng[2]})\n')
                    f.write(f'    Relevance: {len(blocks)} blocks\n')
                    f.write(f'    """\n')
                    
                    # Handle the snippet content
                    snippet_lines = snippet.split('\n')
                    if any(line.strip().startswith(('def ', 'class ')) for line in snippet_lines):
                        # It's already a function/class, add it with proper indentation
                        for line in snippet_lines:
                            if line.strip():
                                f.write(f"    {line}\n")
                            else:
                                f.write("\n")
                    else:
                        # Regular code, just indent it
                        for line in snippet_lines:
                            if line.strip():
                                f.write(f"    {line}\n")
                            else:
                                f.write("\n")
                    f.write("\n")
                
                # Create entity-level function that runs all snippets for this entity
                entity_function = f"run_all_{entity_safe}_snippets"
                entity_functions.append((entity_function, entity, len(blocks)))
                f.write(f"def {entity_function}():\n")
                f.write(f'    """Run all snippets for entity: {entity}"""\n')
                f.write(f"    print(f'Running {len(entity_snippet_functions)} snippets for {entity}')\n")
                for func_name in entity_snippet_functions:
                    f.write(f"    # {func_name}()  # Uncomment to run\n")
                f.write("    pass\n\n")
            
            # Main execution section
            f.write("# " + "="*80 + "\n")
            f.write("# MAIN EXECUTION AND UTILITIES\n")
            f.write("# " + "="*80 + "\n\n")
            
            f.write("def run_top_entities(top_n=3):\n")
            f.write('    """Run snippets for top N most relevant entities."""\n')
            f.write("    top_entities = ENTITY_RANKINGS[:top_n]\n")
            f.write("    for rank, entity, block_count in top_entities:\n")
            f.write("        print(f'=== Rank {rank}: {entity} ({block_count} blocks) ===')\n")
            f.write("        entity_safe = entity.replace('.', '_').replace('-', '_')\n")
            f.write("        func_name = f'run_all_{entity_safe}_snippets'\n")
            f.write("        if func_name in globals():\n")
            f.write("            globals()[func_name]()\n")
            f.write("        print()\n\n")
            
            f.write("def list_all_functions():\n")
            f.write('    """List all available snippet functions."""\n')
            f.write("    functions = [name for name in globals() if name.startswith('rank_')]\n")
            f.write("    functions.sort()\n")
            f.write("    print(f'Available snippet functions ({len(functions)} total):')\n")
            f.write("    for func in functions:\n")
            f.write("        print(f'  - {func}')\n")
            f.write("    return functions\n\n")
            
            f.write("if __name__ == '__main__':\n")
            f.write("    print('Consolidated Code Snippets')\n")
            f.write("    print('=' * 50)\n")
            f.write("    \n")
            f.write("    info = get_entity_info()\n")
            f.write("    print(f'Project: {info[\"project\"]}')\n")
            f.write(f"    print(f'Total entities: {{info[\"total_entities\"]}}')\n")
            f.write(f"    print(f'Total snippets: {snippet_counter}')\n")
            f.write("    print()\n")
            f.write("    \n")
            f.write("    print('Top 5 entities by relevance:')\n")
            f.write("    for rank, entity, count in ENTITY_RANKINGS[:5]:\n")
            f.write("        print(f'{rank}. {entity} ({count} blocks)')\n")
            f.write("    print()\n")
            f.write("    \n")
            f.write("    print('Usage examples:')\n")
            f.write("    print('  run_top_entities(3)      # Run top 3 entities')\n")
            f.write("    print('  list_all_functions()     # List all snippet functions')\n")
            for i, (func_name, entity, _) in enumerate(entity_functions[:3]):
                f.write(f"    print('  {func_name}()  # {entity}')\n")
            f.write("    print()\n")
        
        return {
            'markdown': str(consolidated_md),
            'python': str(consolidated_py)
        }
    
    def process_matched_blocks(self, matched_blocks_file: str = "matched_blocks_ranked.json") -> Dict[str, Dict[str, str]]:
        """
        Process all matched blocks from JSON file and generate output files.
        
        Args:
            matched_blocks_file (str): Path to the matched blocks JSON file
            
        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping entity names to file paths dict
        """
        # Load matched blocks
        try:
            with open(matched_blocks_file, 'r', encoding='utf-8') as f:
                matched_blocks = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {matched_blocks_file} not found!")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON file: {e}")
            return {}
        
        print(f"📁 Loaded matched blocks from: {matched_blocks_file}")
        print(f"📊 Found {len(matched_blocks)} entities to process")
        
        output_files = {}
        
        for entity, blocks in matched_blocks.items():
            snippets, skipped = self.process_entity_blocks(entity, blocks)
            
            if snippets:
                file_paths = self.save_entity_snippets(entity, snippets)
                output_files[entity] = file_paths
                print(f"✅ Saved {len(snippets)} snippets:")
                print(f"   📄 Markdown: `{file_paths['markdown']}`")
                print(f"   🐍 Python: `{file_paths['python']}`")
                print(f"   (Skipped {skipped} invalid blocks)")
            else:
                print(f"⚠ No valid snippets found for entity: {entity}")
        
        print(f"\n🎉 Processing complete! Generated {len(output_files)*2} output files ({len(output_files)} entities).")
        
        # Generate consolidated files with all snippets ranked by entity
        if output_files:
            consolidated_files = self.create_consolidated_files(matched_blocks, output_files)
            print(f"📋 Additional consolidated files:")
            print(f"   📄 All snippets (MD): `{consolidated_files['markdown']}`")
            print(f"   🐍 All snippets (PY): `{consolidated_files['python']}`")
        
        return output_files


def main():
    """Main function to run the code snippet extraction."""
    print("=" * 60)
    print("CODE SNIPPET EXTRACTOR")
    print("=" * 60)
    
    # Initialize extractor
    extractor = CodeSnippetExtractor()
    
    # Process matched blocks
    output_files = extractor.process_matched_blocks()
    
    if output_files:
        print("\n📋 Generated files:")
        for entity, file_paths in output_files.items():
            print(f"  📁 {entity}:")
            print(f"    📄 Markdown: {file_paths['markdown']}")
            print(f"    🐍 Python: {file_paths['python']}")
        
        print(f"\n🔧 Usage Examples:")
        print("# Import and use the generated Python files:")
        for entity in list(output_files.keys())[:2]:  # Show first 2 examples
            entity_safe = entity.replace('.', '_').replace('-', '_')
            print(f"# from outputs.{entity_safe}_snippets import *")
        print("# python outputs/{entity_name}_snippets.py")
    else:
        print("\n❌ No output files generated.")


if __name__ == "__main__":
    main()