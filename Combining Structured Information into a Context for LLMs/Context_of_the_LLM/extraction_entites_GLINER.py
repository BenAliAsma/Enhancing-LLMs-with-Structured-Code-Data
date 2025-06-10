from src.config import commit, date, version, repo_name, problem_stmt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from gliner import GLiNER

class GLiNEREntityExtractor:
    def __init__(self):
        # Initialize the GLiNER model
        self.model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        # Define the labels we want to extract
        self.labels = ["function", "class", "path", "example", "constraint", "module", "condition"]

    def extract_entities(self, text):
        """
        Extract entities using only the GLiNER model
        """
        # Use the model to predict entities
        entities = self.model.predict_entities(text, self.labels, threshold=0.5)

        # Sort entities by their start position
        entities = sorted(entities, key=lambda x: x['start'])

        return entities

    def visualize_results(self, entities, output_file="gliner_extraction_results.png"):
        """
        Visualize the results from GLiNER extraction
        """
        if not entities:
            print("No entities to visualize")
            return {}

        # Prepare data for visualization
        # Count entities by type
        label_counts = Counter([entity['label'] for entity in entities])

        # Calculate confidence distribution
        confidences = {}
        for label in self.labels:
            label_confs = [entity.get('score', 0) for entity in entities if entity['label'] == label]
            if label_confs:
                confidences[label] = {
                    'min': min(label_confs),
                    'max': max(label_confs),
                    'mean': sum(label_confs) / len(label_confs),
                    'count': len(label_confs)
                }

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot entity counts
        df_counts = pd.DataFrame.from_dict(label_counts, orient='index', columns=['count'])
        df_counts = df_counts.reset_index().rename(columns={'index': 'entity_type'})
        df_counts = df_counts.sort_values('count', ascending=False)

        bars = ax1.bar(df_counts['entity_type'], df_counts['count'], color='lightgreen')
        ax1.set_xlabel('Entity Type')
        ax1.set_ylabel('Count')
        ax1.set_title('Entity Types Extracted by GLiNER')
        ax1.tick_params(axis='x', rotation=45)

        # Add counts on top of bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')

        # Plot confidence scores by entity type
        if confidences:
            conf_data = []
            for label, values in confidences.items():
                conf_data.append({
                    'entity_type': label,
                    'confidence': values['mean'],
                    'count': values['count']
                })

            df_conf = pd.DataFrame(conf_data)
            if not df_conf.empty:
                # Size points by count
                sizes = df_conf['count'] * 20
                scatter = ax2.scatter(df_conf['entity_type'], df_conf['confidence'],
                                     s=sizes, alpha=0.7, c=range(len(df_conf)), cmap='viridis')

                # Add a color bar
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Entity Type Index')

                ax2.set_xlabel('Entity Type')
                ax2.set_ylabel('Mean Confidence Score')
                ax2.set_title('GLiNER Confidence by Entity Type')
                ax2.tick_params(axis='x', rotation=45)
                ax2.set_ylim(0, 1.05)

                # Add count labels
                for i, row in df_conf.iterrows():
                    ax2.annotate(f"{row['count']}",
                                (row['entity_type'], row['confidence']),
                                xytext=(0, 10), textcoords='offset points',
                                ha='center')

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Visualization saved to {output_file}")

        return label_counts


# Example usage
if __name__ == "__main__":

    extractor = GLiNEREntityExtractor()
    entities = extractor.extract_entities(problem_stmt)

    print(f"{'Entity':<25} | {'Type':<10} | {'Confidence':<10}")
    print("-" * 50)
    for entity in entities:
        confidence = entity.get('score', 0)
        print(f"{entity['text'][:25]:<25} | {entity['label']:<10} | {confidence:.4f}")

    # Visualize the results
    entity_counts = extractor.visualize_results(entities)
    print("\nEntity type distribution:", dict(entity_counts))