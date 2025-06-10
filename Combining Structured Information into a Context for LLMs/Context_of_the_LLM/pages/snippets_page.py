import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path

# Layout for the snippets page
layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📄 Code Snippets Analysis", className="text-center mb-4"),
            html.Hr(),
            
            # Control panel
            dbc.Card([
                dbc.CardBody([
                    html.H5("Filter Options", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Top N Entities:"),
                            dcc.Slider(
                                id='top-entities-slider',
                                min=5,
                                max=50,
                                step=5,
                                value=20,
                                marks={i: str(i) for i in range(5, 51, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], md=6),
                        dbc.Col([
                            html.Label("Visualization Type:"),
                            dcc.Dropdown(
                                id='viz-type-dropdown',
                                options=[
                                    {'label': 'Entity Size Distribution', 'value': 'entity_size'},
                                    {'label': 'Snippet Count Distribution', 'value': 'snippet_count'},
                                    {'label': 'File Frequency Analysis', 'value': 'file_freq'},
                                    {'label': 'Module Distribution', 'value': 'module_dist'},
                                    {'label': 'Snippet Size Distribution', 'value': 'snippet_size_dist'},
                                    {'label': 'Entity-File Heatmap', 'value': 'heatmap'},
                                    {'label': 'Average Snippet Size', 'value': 'avg_size'}
                                ],
                                value='entity_size'
                            )
                        ], md=6)
                    ])
                ])
            ], className="mb-4"),
            
            # Main visualization
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(id='main-snippet-chart'),
                        type="default"
                    )
                ])
            ], className="mb-4"),
            
            # Statistics cards
            dbc.Row(id='stats-cards', className="mb-4"),
            
            # Detailed analysis
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Detailed Analysis", className="card-title"),
                    html.Div(id='detailed-analysis')
                ])
            ])
            
        ], width=12)
    ])
], fluid=True)

# Callback for main chart
@callback(
    [Output('main-snippet-chart', 'figure'),
     Output('stats-cards', 'children'),
     Output('detailed-analysis', 'children')],
    [Input('top-entities-slider', 'value'),
     Input('viz-type-dropdown', 'value')]
)
def update_snippets_visualization(top_n, viz_type):
    try:
        # Load and process data
        snippet_data = {}
        entity_sizes = {}
        file_frequency = Counter()
        module_frequency = Counter()
        entity_connections = defaultdict(set)
        
        # Process all entity files in outputs directory
        if not os.path.exists("outputs"):
            return {}, [], html.Div("No output files found. Please run the snippet extraction first.")
        
        for filename in os.listdir("outputs"):
            if not filename.endswith(".md"):
                continue

            entity_name = filename.replace("_", ".")[:-3]  # Restore entity name
            entity_sizes[entity_name] = 0
            snippet_data[entity_name] = []

            try:
                with open(os.path.join("outputs", filename), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract snippet sections
                    sections = content.split("## Snippet ")
                    for section in sections[1:]:  # Skip the header
                        try:
                            # Extract file path
                            path_line = section.split('\n')[0]
                            file_path = path_line.split('`')[1]

                            # Count file frequency
                            file_frequency[file_path] += 1

                            # Count module frequency (directory structure)
                            module_path = Path(file_path).parent
                            module_frequency[str(module_path)] += 1

                            # Store connections between entities and files
                            entity_connections[entity_name].add(file_path)

                            # Extract snippet content
                            snippet_content = section.split('```python\n')[1].split('\n```')[0]

                            # Store snippet info
                            snippet_data[entity_name].append({
                                'path': file_path,
                                'content': snippet_content,
                                'size': len(snippet_content)
                            })

                            # Add to entity size
                            entity_sizes[entity_name] += len(snippet_content)
                        except Exception as e:
                            continue
            except Exception as e:
                continue
        
        if not entity_sizes:
            return {}, [], html.Div("No data found to visualize.")
        
        # Generate visualization based on type
        fig = go.Figure()
        
        if viz_type == 'entity_size':
            # Entity size distribution
            sorted_entities = sorted(entity_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n]
            entities = [x[0] for x in sorted_entities]
            sizes = [x[1] for x in sorted_entities]
            
            fig = px.bar(
                x=entities, y=sizes,
                title=f'Top {top_n} Entities by Total Code Size',
                labels={'x': 'Entity', 'y': 'Total Code Size (characters)'},
                color=sizes,
                color_continuous_scale='Blues'
            )
            fig.update_xaxes(tickangle=45)
            
        elif viz_type == 'snippet_count':
            # Snippet count distribution
            snippet_counts = {entity: len(snippets) for entity, snippets in snippet_data.items()}
            sorted_counts = sorted(snippet_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            entities = [x[0] for x in sorted_counts]
            counts = [x[1] for x in sorted_counts]
            
            fig = px.bar(
                x=entities, y=counts,
                title=f'Top {top_n} Entities by Snippet Count',
                labels={'x': 'Entity', 'y': 'Number of Snippets'},
                color=counts,
                color_continuous_scale='Greens'
            )
            fig.update_xaxes(tickangle=45)
            
        elif viz_type == 'file_freq':
            # File frequency analysis
            most_common_files = file_frequency.most_common(top_n)
            files = [f[0] for f in most_common_files]
            counts = [f[1] for f in most_common_files]
            
            fig = px.bar(
                x=counts, y=files,
                orientation='h',
                title=f'Top {top_n} Most Frequent Files',
                labels={'x': 'Number of Snippets', 'y': 'File Path'},
                color=counts,
                color_continuous_scale='Reds'
            )
            
        elif viz_type == 'module_dist':
            # Module distribution
            most_common_modules = module_frequency.most_common(top_n)
            modules = [m[0] for m in most_common_modules]
            counts = [m[1] for m in most_common_modules]
            
            fig = px.pie(
                values=counts, names=modules,
                title=f'Distribution of Snippets by Module (Top {top_n})'
            )
            
        elif viz_type == 'snippet_size_dist':
            # Snippet size distribution
            all_sizes = []
            for entity, snippets in snippet_data.items():
                for snippet in snippets:
                    all_sizes.append(snippet['size'])
            
            fig = px.histogram(
                x=all_sizes,
                nbins=30,
                title='Distribution of Snippet Sizes',
                labels={'x': 'Snippet Size (characters)', 'y': 'Frequency'}
            )
            
        elif viz_type == 'heatmap':
            # Entity-File connection heatmap
            top_entities_list = sorted(entity_sizes.items(), key=lambda x: x[1], reverse=True)[:15]
            top_entity_names = [e[0] for e in top_entities_list]
            
            # Get top files
            all_files = Counter()
            for entity in top_entity_names:
                for file in entity_connections[entity]:
                    all_files[file] += 1
            
            top_files = [file for file, _ in all_files.most_common(15)]
            
            # Create connection matrix
            connection_matrix = np.zeros((len(top_entity_names), len(top_files)))
            for i, entity in enumerate(top_entity_names):
                for j, file in enumerate(top_files):
                    if file in entity_connections[entity]:
                        connection_matrix[i, j] = 1
            
            fig = px.imshow(
                connection_matrix,
                x=top_files,
                y=top_entity_names,
                title='Entity-File Connection Heatmap',
                labels={'x': 'Files', 'y': 'Entities', 'color': 'Connected'},
                color_continuous_scale='Blues'
            )
            fig.update_xaxes(tickangle=45)
            
        elif viz_type == 'avg_size':
            # Average snippet size by entity
            avg_sizes = {}
            for entity, snippets in snippet_data.items():
                if snippets:
                    avg_sizes[entity] = sum(s['size'] for s in snippets) / len(snippets)
            
            sorted_avg = sorted(avg_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n]
            entities = [x[0] for x in sorted_avg]
            sizes = [x[1] for x in sorted_avg]
            
            fig = px.bar(
                x=entities, y=sizes,
                title=f'Top {top_n} Entities by Average Snippet Size',
                labels={'x': 'Entity', 'y': 'Average Snippet Size (characters)'},
                color=sizes,
                color_continuous_scale='Purples'
            )
            fig.update_xaxes(tickangle=45)
        
        # Update layout
        fig.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=60, b=100),
            font_size=12
        )
        
        # Generate statistics cards
        total_entities = len(entity_sizes)
        total_snippets = sum(len(snippets) for snippets in snippet_data.values())
        total_files = len(file_frequency)
        total_modules = len(module_frequency)
        
        all_sizes = []
        for entity, snippets in snippet_data.items():
            for snippet in snippets:
                all_sizes.append(snippet['size'])
        
        avg_snippet_size = sum(all_sizes) / len(all_sizes) if all_sizes else 0
        
        stats_cards = [
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(total_entities), className="text-primary"),
                        html.P("Total Entities", className="text-muted")
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(total_snippets), className="text-success"),
                        html.P("Total Snippets", className="text-muted")
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(total_files), className="text-info"),
                        html.P("Unique Files", className="text-muted")
                    ])
                ])
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{avg_snippet_size:.0f}", className="text-warning"),
                        html.P("Avg. Snippet Size", className="text-muted")
                    ])
                ])
            ], md=3)
        ]
        
        # Generate detailed analysis
        top_entities_list = sorted(entity_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_files_list = file_frequency.most_common(10)
        
        detailed_analysis = html.Div([
            dbc.Row([
                dbc.Col([
                    html.H6("🏆 Top 10 Entities by Size"),
                    html.Ol([
                        html.Li(f"{entity}: {size:,} chars ({len(snippet_data[entity])} snippets)")
                        for entity, size in top_entities_list
                    ])
                ], md=6),
                dbc.Col([
                    html.H6("📁 Top 10 Most Referenced Files"),
                    html.Ol([
                        html.Li(f"{file}: {count} references")
                        for file, count in top_files_list
                    ])
                ], md=6)
            ])
        ])
        
        return fig, stats_cards, detailed_analysis
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error loading data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        error_fig.update_layout(title="Error Loading Snippets Data")
        
        return error_fig, [], html.Div(f"Error: {str(e)}")