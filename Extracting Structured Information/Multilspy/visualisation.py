import json
import networkx as nx
import plotly.graph_objects as go

def build_graph(data, graph, parent=None, prefix='root', depth=0, max_depth=3):
    if depth > max_depth:
        return
    if isinstance(data, dict):
        for key, value in data.items():
            node_id = f"{prefix}.{key}"
            graph.add_node(node_id, label=key)
            if parent:
                graph.add_edge(parent, node_id)
            build_graph(value, graph, node_id, node_id, depth+1, max_depth)
    elif isinstance(data, list):
        for i, item in enumerate(data[:10]):  # limite aussi les très grandes listes
            node_id = f"{prefix}[{i}]"
            graph.add_node(node_id, label=f"[{i}]")
            if parent:
                graph.add_edge(parent, node_id)
            build_graph(item, graph, node_id, node_id, depth+1, max_depth)
    else:
        val_node = f"{prefix}.val"
        graph.add_node(val_node, label=str(data))
        if parent:
            graph.add_edge(parent, val_node)


def visualize_interactive(json_data):
    G = nx.DiGraph()
    build_graph(json_data, G, max_depth=3)
    pos = nx.spring_layout(G, k=0.5, iterations=100)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
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
        node_text.append(G.nodes[node]['label'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='skyblue',
            size=20,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
               layout=go.Layout(
                   title=dict(
                       text='JSON Structure Visualization',
                       font=dict(size=16)
                   ),
                   showlegend=False,
                   hovermode='closest',
                   margin=dict(b=20, l=5, r=5, t=40),
                   xaxis=dict(showgrid=False, zeroline=False),
                   yaxis=dict(showgrid=False, zeroline=False)
               ))

    fig.write_html("json_tree_view.html", auto_open=True)
    fig.show()

# Exemple d'utilisation
if __name__ == "__main__":
    with open("\\home\\abenali\\Enhancing-LLMs-with-Structured-Code-Data\\Extracting Structured Information\\Multilspy\\project_metadata_multi.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    visualize_interactive(data)

