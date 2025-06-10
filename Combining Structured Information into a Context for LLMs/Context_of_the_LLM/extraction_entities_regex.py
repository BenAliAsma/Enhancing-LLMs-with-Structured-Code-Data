from src.config import commit, date, version, repo_name, problem_stmt
import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

class RegexEntityExtractor:
    def __init__(self):
        self.patterns = {
        "function": r'(?<!\.)\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()',
        "class": r'\b(?:class\s+|new\s+|m\.|models\.)([A-Z][a-zA-Z0-9_]*)\b',
        "variable": r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?==(?!=))',
        "path": r'(?:from\s+|import\s+)([\w\.]+)(?:\s+import\s+[\w\., ]+)?|https://github\.com/[^\s\'"]+',
        "example": r'(>>>.*?)(?=\n(?!\.\.\.|>>>)|$)',
        "module": r'\b(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
        "condition": r'\b(if|while|for|try|except|finally|assert)\b'
        }

    def extract_entities(self, text):
        entities = []

        # Extract entities from the entire text
        for label, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.DOTALL):
                for i, group in enumerate(match.groups() if match.groups() else (match.group(),)):
                    if group and i % 2 == 0:
                        entities.append({
                            "text": group.strip(),
                            "label": label,
                            "start": match.start(),
                            "end": match.end()
                        })

        # Special handling for code blocks
        code_blocks = list(re.finditer(r'```(?:python)?(.*?)```', text, re.DOTALL | re.IGNORECASE))
        for block in code_blocks:
            code = block.group(1)
            for label, pattern in self.patterns.items():
                for match in re.finditer(pattern, code):
                    entities.append({
                        "text": match.group().strip(),
                        "label": label,
                        "start": block.start() + match.start(),
                        "end": block.start() + match.end()
                    })

        return entities

    def visualize_results(self, entities, output_file="regex_extraction_results.png"):
        # Count entity types
        label_counts = Counter([entity['label'] for entity in entities])

        # Create DataFrame for visualization
        df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['count'])
        df = df.reset_index().rename(columns={'index': 'entity_type'})
        df = df.sort_values('count', ascending=False)

        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['entity_type'], df['count'], color='skyblue')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.title('Entity Types Extracted by Regex Method')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Add counts on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')

        plt.savefig(output_file)
        plt.close()
        print(f"Visualization saved to {output_file}")

        return label_counts


# Example usage
if __name__ == "__main__":

    extractor = RegexEntityExtractor()
    entities = extractor.extract_entities(problem_stmt)

    print(f"{'Entity':<25} | {'Type':<10}")
    print("-" * 40)
    for entity in entities:
        print(f"{entity['text']:<25} | {entity['label']:<10}")

    # Visualize the results
    entity_counts = extractor.visualize_results(entities)
    print("\nEntity type distribution:", dict(entity_counts))