import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from pages import snippets_page, ranked_entities_page

# Initialize the Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True,
                use_pages=False)

# Define the main layout with navigation
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Navigation bar
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")),
            dbc.NavItem(dbc.NavLink("Code Snippets", href="/snippets", active="exact")),
            dbc.NavItem(dbc.NavLink("Ranked Entities", href="/ranked_entities", active="exact")),
        ],
        brand="Code Analysis Dashboard",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    # Page content
    html.Div(id='page-content')
])

# Home page layout
def home_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Code Analysis Dashboard", className="text-center mb-4"),
                html.Hr(),
                
                # Welcome section
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Welcome to the Code Analysis Dashboard", className="card-title"),
                        html.P([
                            "This dashboard provides comprehensive visualization and analysis of your codebase. ",
                            "Navigate through different sections to explore various aspects of your code structure and entity relationships."
                        ], className="card-text"),
                    ])
                ], className="mb-4"),
                
                # Dashboard sections
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("📄 Code Snippets Analysis", className="card-title"),
                                html.P([
                                    "Explore extracted code snippets from your repository. ",
                                    "View distribution patterns, file frequency analysis, and entity-file relationships."
                                ], className="card-text"),
                                dbc.Button("View Snippets", href="/snippets", color="primary", className="mt-2")
                            ])
                        ])
                    ], md=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("📊 Ranked Entities", className="card-title"),
                                html.P([
                                    "Analyze entity rankings and their relationships. ",
                                    "Discover patterns in entity importance and code structure hierarchies."
                                ], className="card-text"),
                                dbc.Button("View Rankings", href="/ranked_entities", color="success", className="mt-2")
                            ])
                        ])
                    ], md=6),
                ], className="mb-4"),
                
                # Features section
                dbc.Card([
                    dbc.CardBody([
                        html.H4("🚀 Dashboard Features", className="card-title"),
                        html.Ul([
                            html.Li("Interactive visualizations with Plotly"),
                            html.Li("Real-time data filtering and analysis"),
                            html.Li("Entity relationship mapping"),
                            html.Li("Code structure insights"),
                            html.Li("File and module frequency analysis"),
                            html.Li("Snippet size and distribution metrics"),
                        ])
                    ])
                ], className="mb-4"),
                
                # Statistics overview
                html.Div(id="stats-overview")
                
            ], width=12)
        ])
    ], fluid=True)

# Callback for page routing
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/snippets':
        return snippets_page.layout
    elif pathname == '/ranked_entities':
        return ranked_entities_page.layout
    else:  # Default to home page
        return home_layout()

# Callback to load statistics on home page
@app.callback(
    Output('stats-overview', 'children'),
    Input('url', 'pathname')
)
def load_stats(pathname):
    if pathname == '/':
        try:
            import json
            import os
            from pathlib import Path
            
            stats = {}
            
            # Try to load matched blocks
            if os.path.exists('matched_blocks_ranked.json'):
                with open('matched_blocks_ranked.json', 'r') as f:
                    matched_blocks = json.load(f)
                    stats['total_entities'] = len(matched_blocks)
                    stats['total_blocks'] = sum(len(blocks) for blocks in matched_blocks.values())
            
            # Count output files
            if os.path.exists('outputs'):
                md_files = len([f for f in os.listdir('outputs') if f.endswith('.md')])
                py_files = len([f for f in os.listdir('outputs') if f.endswith('.py')])
                stats['markdown_files'] = md_files
                stats['python_files'] = py_files
            
            # Count visualization files
            if os.path.exists('visualizationsnippet'):
                viz_files = len([f for f in os.listdir('visualizationsnippet') if f.endswith('.png')])
                stats['visualization_files'] = viz_files
            
            if stats:
                return dbc.Card([
                    dbc.CardBody([
                        html.H4("📈 Quick Statistics", className="card-title"),
                        dbc.Row([
                            dbc.Col([
                                html.H5(stats.get('total_entities', 'N/A'), className="text-primary"),
                                html.P("Total Entities", className="text-muted")
                            ], md=3),
                            dbc.Col([
                                html.H5(stats.get('total_blocks', 'N/A'), className="text-success"),
                                html.P("Code Blocks", className="text-muted")
                            ], md=3),
                            dbc.Col([
                                html.H5(stats.get('markdown_files', 'N/A'), className="text-info"),
                                html.P("Generated MD Files", className="text-muted")
                            ], md=3),
                            dbc.Col([
                                html.H5(stats.get('visualization_files', 'N/A'), className="text-warning"),
                                html.P("Visualizations", className="text-muted")
                            ], md=3),
                        ])
                    ])
                ])
            
        except Exception as e:
            return dbc.Alert(f"Could not load statistics: {str(e)}", color="warning")
    
    return html.Div()

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8050)