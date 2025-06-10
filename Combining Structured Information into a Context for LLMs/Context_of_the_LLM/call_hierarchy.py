import shutil
import os
import ast
from pathlib import Path
from collections import defaultdict

class AstropyCallHierarchyAnalyzer:
    def __init__(self, project_path="astropy"):
        self.project_path = Path(project_path)
        self.call_graph = defaultdict(set)
        self.function_definitions = set()
        self.all_hierarchy = []

    def analyze_project(self):
        """Main method to analyze the entire astropy project"""
        print("Starting astropy call hierarchy analysis...")

        # Analyze all Python files
        python_files = list(self.project_path.rglob("*.py"))
        total_files = len(python_files)

        print(f"Found {total_files} Python files to analyze")

        for i, file_path in enumerate(python_files):
            if i % 100 == 0:
                print(f"Progress: {i}/{total_files} files analyzed")

            try:
                self._analyze_file(file_path)
            except Exception as e:
                continue

        print("Building call hierarchy chains...")
        self._build_all_hierarchy()
        print("Analysis complete!")

    def _analyze_file(self, file_path):
        """Analyze a single Python file for function calls and definitions"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            relative_path = file_path.relative_to(self.project_path)
            module_name = self._path_to_module_name(relative_path)

            visitor = CallHierarchyVisitor(module_name, self)
            visitor.visit(tree)

        except (SyntaxError, UnicodeDecodeError):
            pass

    def _path_to_module_name(self, path):
        """Convert file path to module name"""
        parts = list(path.parts[:-1]) + [path.stem]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else ""

    def _build_all_hierarchy(self):
        """Build all call hierarchy chains"""
        # Find all possible call chains
        visited_chains = set()

        for func in self.function_definitions:
            chains = self._get_call_chains_from_function(func, max_depth=5)
            for chain in chains:
                chain_str = " -> ".join(chain)
                if len(chain) > 1 and chain_str not in visited_chains:
                    self.all_hierarchy.append(chain_str)
                    visited_chains.add(chain_str)

    def _get_call_chains_from_function(self, start_func, max_depth=5, current_path=None, visited=None):
        """Get all call chains starting from a function"""
        if current_path is None:
            current_path = [start_func]
        if visited is None:
            visited = set()

        # Avoid infinite recursion
        if start_func in visited or len(current_path) >= max_depth:
            return [current_path]

        visited.add(start_func)
        chains = []

        # If this function calls others, extend the chain
        if start_func in self.call_graph and self.call_graph[start_func]:
            for called_func in self.call_graph[start_func]:
                new_path = current_path + [called_func]
                sub_chains = self._get_call_chains_from_function(
                    called_func, max_depth, new_path, visited.copy()
                )
                chains.extend(sub_chains)
        else:
            # End of chain
            chains.append(current_path)

        return chains

    def print_all_hierarchy(self):
        """Print all hierarchy in the requested format"""
        if not self.all_hierarchy:
            print("No call hierarchies found.")
            return

        print("\nAll Call Hierarchies:")
        print("-" * 80)
        for hierarchy in sorted(self.all_hierarchy):
            print(f"        {hierarchy}")

        print(f"\nTotal hierarchies found: {len(self.all_hierarchy)}")

    def get_all_hierarchy(self):
        """Return the all_hierarchy list for external access"""
        return self.all_hierarchy


class CallHierarchyVisitor(ast.NodeVisitor):
    def __init__(self, module_name, analyzer):
        self.module_name = module_name
        self.analyzer = analyzer
        self.current_function = None
        self.imports = {}
        self.local_functions = set()

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[-1]
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if node.module == "." or node.module.startswith("."):
                    # Relative import
                    full_name = f"{self.module_name}.{alias.name}"
                else:
                    full_name = f"{node.module}.{alias.name}"
                self.imports[name] = full_name
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Create function identifier
        if self.module_name:
            func_name = f"{self.module_name}.{node.name}"
        else:
            func_name = node.name

        # Clean up function name for display
        func_display_name = self._clean_function_name(func_name)

        self.analyzer.function_definitions.add(func_display_name)
        self.local_functions.add(node.name)

        old_function = self.current_function
        self.current_function = func_display_name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        # Handle async functions the same way
        self.visit_FunctionDef(node)

    def visit_Call(self, node):
        if self.current_function:
            called_name = self._resolve_call(node)
            if called_name:
                clean_called_name = self._clean_function_name(called_name)
                self.analyzer.call_graph[self.current_function].add(clean_called_name)
        self.generic_visit(node)

    def _resolve_call(self, node):
        """Resolve a function call to its name"""
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.imports:
                return self.imports[name]
            elif name in self.local_functions:
                return f"{self.module_name}.{name}" if self.module_name else name
            else:
                return name

        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                base_name = node.func.value.id
                attr_name = node.func.attr

                if base_name in self.imports:
                    return f"{self.imports[base_name]}.{attr_name}"
                elif base_name == "self":
                    return f"{self.module_name}.{attr_name}" if self.module_name else attr_name
                else:
                    return f"{base_name}.{attr_name}"

        return None

    def _clean_function_name(self, name):
        """Clean function name for better display"""
        # Remove common prefixes that make names too long
        if name.startswith("astropy."):
            name = name[8:]  # Remove "astropy." prefix

        # Simplify very long module paths
        parts = name.split(".")
        if len(parts) > 3:
            # Keep first part, last two parts
            name = f"{parts[0]}.....{parts[-2]}.{parts[-1]}"

        return name


def analyze_astropy_project(project_path="astropy"):
    """
    Analyze astropy project and return the analyzer instance.
    This function can be imported and used from other files.
    """
    # Setup - Clone astropy if not exists
    if not os.path.exists(project_path):
        print(f"Cloning astropy repository to {project_path}...")
        os.system(f"git clone https://github.com/astropy/astropy.git {project_path}")
        
    print("Current working directory:", os.getcwd())

    # Initialize analyzer
    analyzer = AstropyCallHierarchyAnalyzer(project_path)

    # Run analysis
    analyzer.analyze_project()

    return analyzer


def main():
    """Main function to run the call hierarchy analysis"""
    # Run analysis
    analyzer = analyze_astropy_project()

    # Print results in the requested format
    analyzer.print_all_hierarchy()

    # Store in variable as requested
    call_hierarchy = analyzer.get_all_hierarchy()

    print(f"\nVariable 'call_hierarchy' contains {len(call_hierarchy)} call chains")
    
    return analyzer, call_hierarchy


# Global variables that can be accessed from other files
analyzer_instance = None
all_hierarchy = []

def get_analyzer():
    """Get the global analyzer instance"""
    global analyzer_instance
    if analyzer_instance is None:
        analyzer_instance = analyze_astropy_project()
    return analyzer_instance

def get_all_hierarchy():
    """Get the all_hierarchy list"""
    global all_hierarchy
    if not all_hierarchy:
        analyzer = get_analyzer()
        all_hierarchy = analyzer.get_all_hierarchy()
    return all_hierarchy


if __name__ == "__main__":
    analyzer_instance, all_hierarchy = main()