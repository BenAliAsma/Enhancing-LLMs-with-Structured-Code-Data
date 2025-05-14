import json
import networkx as nx
import plotly.graph_objects as go
import dash
from dash import dcc, html
import pandas as pd

# Load LSIF data
def load_lsif_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return pd.DataFrame(data)

# Process LSIF vertices and edges
def process_lsif_data(df):
    vertices = df[df['type'] == 'vertex']
    edges = df[df['type'] == 'edge']
    return vertices, edges

# Create a graph from vertices and edges
def create_graph(vertices, edges):
    G = nx.DiGraph()

    # Add nodes
    for _, v in vertices.iterrows():
        G.add_node(v['id'], label=v['label'])

    # Add edges
    for _, e in edges.iterrows():
        if 'inVs' in e and isinstance(e['inVs'], list):
            # multiple targets
            for target in e['inVs']:
                G.add_edge(e['outV'], target, label=e.get('label', ''))
        elif 'inV' in e:
            # single target
            G.add_edge(e['outV'], e['inV'], label=e.get('label', ''))

    return G

# Visualize graph using Plotly
def plot_graph(G):
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"ID: {node}<br>Label: {G.nodes[node]['label']}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=[len(G.adj[n]) for n in G.nodes()],
            size=10,
            line_width=2
        ),
        text=[G.nodes[node]['label'] for node in G.nodes()],
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text="LSIF Graph (Vertices and Edges)",
                            font=dict(size=20)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="LSIF Metadata Visualization",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig

# Dash dashboard
def create_dashboard(G):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("LSIF Metadata Graph"),
        dcc.Graph(
            id='graph',
            figure=plot_graph(G)
        )
    ])

    app.run(debug=True)

if __name__ == '__main__':
    lsif_file_path = 'exemple.lsif'  
    df = load_lsif_data(lsif_file_path)
    vertices, edges = process_lsif_data(df)
    G = create_graph(vertices, edges)

    create_dashboard(G)
