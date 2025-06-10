import json
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import webbrowser
import threading
import time
from datetime import datetime
import sys
import os
import html as html_escape  # Rename to avoid conflict

# Import from your existing ranking.py
try:
    from src.config import commit, date, version, repo_name, problem_stmt
    from ranking_entities import (
        entities, updated_entities, weights, code_positions, 
        error_positions, auto_query, Entity
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure ranking_entities.py and src/config.py are available in your path")
    sys.exit(1)

class ProfessionalEntityVisualizer:
    def __init__(self):
        """Initialize the Professional Entity Visualizer"""
        self.problem_stmt = problem_stmt
        self.updated_entities = updated_entities
        self.weights = weights
        self.code_positions = code_positions
        self.error_positions = error_positions
        self.auto_query = auto_query
        
        # Enhanced professional color palette matching your matplotlib design
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        # Enhanced entity type colors - matching your matplotlib design
        self.entity_colors = {
            'function': '#F94144',
            'variable': '#F3722C', 
            'path': '#F9C74F',
            'class': '#90BE6D',
            'example': '#577590',
            'module': '#1f77b4',
            'constraint': '#e377c2',
            'condition': '#bcbd22',
            'unknown': '#7f7f7f'
        }
        
        self.processed_entities = self._process_entities()
        self.app = self._create_dash_app()
        
    def _process_entities(self):
        """Process entities to add additional metrics"""
        processed = []
        
        for i, entity in enumerate(self.updated_entities):
            entity_dict = {
                'rank': i + 1,
                'text': entity.text,
                'label': entity.label,
                'start': entity.start,
                'end': entity.end,
                'source': entity.source,
                'confidence': entity.confidence,
                'length': len(entity.text),
                'position_normalized': entity.start / len(self.problem_stmt) if self.problem_stmt else 0,
                'in_code': entity.in_code,  
                'near_error': entity.near_error, 
                'bm25_score': entity.bm25_score,  
                'type_weight': self.weights.get('type', {}).get(entity.label.lower(), 0.0) if isinstance(self.weights, dict) else 0.0
            }
            processed.append(entity_dict)
            
        return processed
    
    def create_confidence_distribution_chart(self):
        """Create confidence distribution chart"""
        confidences = [e['confidence'] for e in self.processed_entities]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=confidences,
            nbinsx=20,
            marker_color=self.colors['primary'],
            opacity=0.7,
            name='Confidence Distribution'
        ))
        
        fig.update_layout(
            title={
                'text': 'Entity Confidence Score Distribution',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Confidence Score',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_enhanced_ranking_chart(self, top_n=15):
        """Create enhanced ranking chart matching matplotlib design"""
        chart_data = self.processed_entities[:top_n]
        
        fig = go.Figure()
        
        # Add confidence bars with enhanced styling
        fig.add_trace(go.Bar(
            x=[e['confidence'] for e in chart_data],
            y=[f"{e['text'][:30]}..." if len(e['text']) > 30 else e['text'] for e in chart_data],
            orientation='h',
            marker=dict(
                color=[self.entity_colors.get(e['label'], self.colors['primary']) for e in chart_data],
                line=dict(color='rgba(255,255,255,0.8)', width=1),
                opacity=0.8
            ),
            text=[f"#{e['rank']} - {e['confidence']:.4f}" for e in chart_data],
            textposition='inside',
            textfont=dict(color='white', size=10, family='Arial Black'),
            name='Confidence Score',
            hovertemplate='<b>%{y}</b><br>' +
                         'Rank: #%{customdata[0]}<br>' +
                         'Confidence: %{x:.4f}<br>' +
                         'Label: %{customdata[1]}<br>' +
                         'Source: %{customdata[2]}<br>' +
                         '<extra></extra>',
            customdata=[[e['rank'], e['label'], e['source']] for e in chart_data]
        ))
        
        fig.update_layout(
            title={
                'text': f'Top {top_n} Entities by Confidence Score',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif', 'color': '#2c3e50'}
            },
            xaxis_title='Confidence Score',
            yaxis_title='Entities',
            template='plotly_white',
            height=max(500, top_n * 35),
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_label_distribution_chart(self):
        """Create label distribution pie chart"""
        label_counts = {}
        for entity in self.processed_entities:
            label = entity['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        fig = go.Figure(data=[go.Pie(
            labels=list(label_counts.keys()),
            values=list(label_counts.values()),
            marker_colors=[self.entity_colors.get(label, self.colors['primary']) for label in label_counts.keys()],
            textinfo='label+percent+value',
            textposition='auto',
            textfont=dict(size=12, family='Arial, sans-serif'),
            hole=0.3
        )])
        
        fig.update_layout(
            title={
                'text': 'Entity Label Distribution',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_metrics_scatter_plot(self):
        """Create scatter plot of confidence vs position"""
        fig = go.Figure()
        
        for label in set(e['label'] for e in self.processed_entities):
            label_data = [e for e in self.processed_entities if e['label'] == label]
            
            fig.add_trace(go.Scatter(
                x=[e['position_normalized'] for e in label_data],
                y=[e['confidence'] for e in label_data],
                mode='markers',
                marker=dict(
                    size=12,
                    color=self.entity_colors.get(label, self.colors['primary']),
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                text=[f"Rank #{e['rank']}: {e['text']}" for e in label_data],
                name=label.title(),
                hovertemplate='<b>%{text}</b><br>Position: %{x:.3f}<br>Confidence: %{y:.4f}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': 'Entity Position vs Confidence Score',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Normalized Position in Text',
            yaxis_title='Confidence Score',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_weighting_system_chart(self):
        """Create visualization of the weighting system"""
        if not isinstance(self.weights, dict):
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Feature Importance Weights', 'Entity Type Weights'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # General weights (excluding 'type')
        general_weights = {k: v for k, v in self.weights.items() if k != 'type' and isinstance(v, (int, float))}
        if general_weights:
            sorted_general = dict(sorted(general_weights.items(), key=lambda x: x[1], reverse=True))
            
            fig.add_trace(
                go.Bar(
                    x=list(sorted_general.keys()),
                    y=list(sorted_general.values()),
                    marker_color=self.colors['info'],
                    text=[f'{v:.2f}' for v in sorted_general.values()],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # Entity type weights
        type_weights = self.weights.get('type', {})
        if type_weights:
            sorted_types = dict(sorted(type_weights.items(), key=lambda x: x[1], reverse=True))
            
            fig.add_trace(
                go.Bar(
                    x=list(sorted_types.keys()),
                    y=list(sorted_types.values()),
                    marker_color=[self.entity_colors.get(t, self.colors['primary']) for t in sorted_types.keys()],
                    text=[f'{v:.2f}' for v in sorted_types.values()],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title={
                'text': 'Entity Ranking Weight System',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_extraction_flow_chart(self):
        """Create a flow chart showing the extraction process"""
        fig = go.Figure()
        
        # Define nodes for the extraction process
        nodes = {
            'Source Text': (0.1, 0.8),
            'Regex Patterns': (0.3, 0.9),
            'GLiNER Model': (0.3, 0.7),
            'Regex Entities': (0.5, 0.9),
            'GLiNER Entities': (0.5, 0.7),
            'Combined Entities': (0.7, 0.8),
            'Feature Extraction': (0.85, 0.8),
            'Final Ranking': (1.0, 0.8)
        }
        
        # Add nodes as scatter points
        for name, (x, y) in nodes.items():
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=30, color=self.colors['primary']),
                text=[name],
                textposition="middle center",
                textfont=dict(color="white", size=10),
                showlegend=False
            ))
        
        # Add arrows (simplified as lines)
        arrows = [
            ('Source Text', 'Regex Patterns'),
            ('Source Text', 'GLiNER Model'),
            ('Regex Patterns', 'Regex Entities'),
            ('GLiNER Model', 'GLiNER Entities'),
            ('Regex Entities', 'Combined Entities'),
            ('GLiNER Entities', 'Combined Entities'),
            ('Combined Entities', 'Feature Extraction'),
            ('Feature Extraction', 'Final Ranking')
        ]
        
        for start, end in arrows:
            x0, y0 = nodes[start]
            x1, y1 = nodes[end]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color=self.colors['secondary'], width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title={
                'text': 'Entity Extraction Process Flow',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_entity_metrics_heatmap(self):
        """Create a heatmap of entity metrics"""
        if not self.processed_entities:
            return go.Figure()
        
        # Select top entities for readability
        top_entities = self.processed_entities[:15]
        
        # Create metrics matrix
        metrics = ['confidence', 'bm25_score', 'type_weight', 'position_normalized']
        entity_names = [e['text'][:20] + ('...' if len(e['text']) > 20 else '') for e in top_entities]
        
        z_values = []
        for metric in metrics:
            values = [e[metric] for e in top_entities]
            # Normalize values to 0-1 range for better visualization
            if max(values) > min(values):
                normalized = [(v - min(values)) / (max(values) - min(values)) for v in values]
            else:
                normalized = values
            z_values.append(normalized)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=entity_names,
            y=metrics,
            colorscale='Viridis',
            text=[[f'{top_entities[j][metrics[i]]:.3f}' for j in range(len(top_entities))] for i in range(len(metrics))],
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title={
                'text': 'Entity Metrics Heatmap (Top 15 Entities)',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Entities',
            yaxis_title='Metrics',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_detailed_table(self):
        """Create detailed data table"""
        table_data = []
        for entity in self.processed_entities:
            table_data.append({
                'Rank': entity['rank'],
                'Entity': entity['text'],
                'Label': entity['label'],
                'Confidence': f"{entity['confidence']:.6f}",
                'BM25 Score': f"{entity['bm25_score']:.3f}",
                'Type Weight': f"{entity['type_weight']:.3f}",
                'In Code': '✓' if entity['in_code'] else '✗',
                'Near Error': '✓' if entity['near_error'] else '✗',
                'Position': f"{entity['position_normalized']:.3f}",
                'Length': entity['length']
            })
        
        return table_data
    
    def _render_highlighted_text_html(self, entities_to_highlight):
        if not self.problem_stmt or not entities_to_highlight:
            return html_escape.escape(self.problem_stmt) if self.problem_stmt else ""
    
        result = ""
        last_idx = 0
    
    # Sort entities by start position and remove overlaps
        sorted_entities = sorted(entities_to_highlight, key=lambda x: x['start'])
        non_overlapping = []
    
        for entity in sorted_entities:
        # Skip if this entity overlaps with the previous one
            if non_overlapping and entity['start'] < non_overlapping[-1]['end']:
                continue
            non_overlapping.append(entity)
    
        for entity in non_overlapping:
            start, end = entity['start'], entity['end']
        
        # Skip if start position is before our current position
            if start < last_idx:
                continue
            
        # Add plain text before entity
            if start > last_idx:
                result += html_escape.escape(self.problem_stmt[last_idx:start])
        
        # Get entity styling
            color = self.entity_colors.get(entity['label'], '#DDD')
            rank = entity['rank']
        
        # Enhanced styling based on rank
            if rank <= 5:
                opacity = '1.0'
                border = '3px solid #2c3e50'
                shadow = 'box-shadow: 0 4px 8px rgba(0,0,0,0.3);'
                font_weight = 'bold'
            elif rank <= 10:
                opacity = '0.9'
                border = '2px solid #34495e'
                shadow = 'box-shadow: 0 2px 4px rgba(0,0,0,0.2);'
                font_weight = 'bold'
            else:
                opacity = str(max(0.4, 1.0 - (rank - 10) * 0.03))
                border = '1px solid rgba(255,255,255,0.5)'
                shadow = ''
                font_weight = 'normal'
        
        # Safely extract entity text
            entity_text = html_escape.escape(str(self.problem_stmt[start:end]))
        
        # Create enhanced span with rank badge
            span = f"""<span style="
                background-color:{color}; 
                opacity:{opacity}; 
                padding:4px 8px; 
                margin:1px; 
                border-radius:6px; 
                color:white; 
                font-weight:{font_weight}; 
                border:{border}; 
                {shadow} 
                display:inline-block; 
                position:relative; 
                font-family:monospace;" 
                title="🏆 Rank: {rank} | 🏷️ {entity['label'].title()} | 📊 Confidence: {entity['confidence']:.4f} | 📍 Position: {entity['start']}-{entity['end']} | 🔍 BM25: {entity.get('bm25_score', 0):.3f}">
                {entity_text}<sup style="background:#e74c3c; color:white; padding:1px 4px; border-radius:8px; font-size:10px; margin-left:2px;">{rank}</sup>
            </span>"""
        
            result += span
            last_idx = end
    
    # Add remaining text
        if last_idx < len(self.problem_stmt):
            result += html_escape.escape(self.problem_stmt[last_idx:])
    
        return result

    def _create_highlighted_text_components(self, entities_to_highlight):
        """Create Dash-compatible highlighted text components"""
        if not self.problem_stmt or not entities_to_highlight:
            return [html.Span(self.problem_stmt if self.problem_stmt else "No text available")]
        
        # Sort entities by start position and remove overlaps
        sorted_entities = sorted(entities_to_highlight, key=lambda x: x['start'])
        non_overlapping = []
        
        for entity in sorted_entities:
            # Skip if this entity overlaps with the previous one
            if non_overlapping and entity['start'] < non_overlapping[-1]['end']:
                continue
            non_overlapping.append(entity)
        
        components = []
        last_idx = 0
        
        for entity in non_overlapping:
            start, end = entity['start'], entity['end']
            
            # Skip if start position is before our current position
            if start < last_idx:
                continue
            
            # Add plain text before entity
            if start > last_idx:
                plain_text = self.problem_stmt[last_idx:start]
                if plain_text.strip():  # Only add if not just whitespace
                    components.append(html.Span(plain_text))
            
            # Get entity styling based on rank
            color = self.entity_colors.get(entity['label'], '#DDD')
            rank = entity['rank']
            
            # Enhanced styling based on rank
            if rank <= 5:
                style = {
                    'backgroundColor': color,
                    'color': 'white',
                    'padding': '4px 8px',
                    'margin': '1px',
                    'borderRadius': '6px',
                    'fontWeight': 'bold',
                    'border': '3px solid #2c3e50',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.3)',
                    'display': 'inline-block',
                    'position': 'relative'
                }
            elif rank <= 10:
                style = {
                    'backgroundColor': color,
                    'color': 'white',
                    'padding': '4px 8px',
                    'margin': '1px',
                    'borderRadius': '6px',
                    'fontWeight': 'bold',
                    'border': '2px solid #34495e',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
                    'display': 'inline-block'
                }
            else:
                opacity = max(0.4, 1.0 - (rank - 10) * 0.03)
                style = {
                    'backgroundColor': color,
                    'color': 'white',
                    'padding': '4px 8px',
                    'margin': '1px',
                    'borderRadius': '6px',
                    'border': '1px solid rgba(255,255,255,0.5)',
                    'display': 'inline-block',
                    'opacity': str(opacity)
                }
            
            # Create the highlighted span with tooltip
            entity_text = str(self.problem_stmt[start:end])
            tooltip_text = f"Rank: {rank} | {entity['label'].title()} | Confidence: {entity['confidence']:.4f} | BM25: {entity.get('bm25_score', 0):.3f}"
            
            components.append(
                html.Span([
                    entity_text,
                    html.Sup(str(rank), style={
                        'backgroundColor': '#e74c3c',
                        'color': 'white',
                        'padding': '1px 4px',
                        'borderRadius': '8px',
                        'fontSize': '10px',
                        'marginLeft': '2px'
                    })
                ], 
                style=style,
                title=tooltip_text
                )
            )
            
            last_idx = end
        
        # Add remaining text
        if last_idx < len(self.problem_stmt):
            remaining_text = self.problem_stmt[last_idx:]
            if remaining_text.strip():  # Only add if not just whitespace
                components.append(html.Span(remaining_text))
        
        return components

    def create_entity_legend(self):
        """Create an enhanced legend for entity types"""
        legend_items = []
        
        for label, color in self.entity_colors.items():
            count = len([e for e in self.processed_entities if e['label'] == label])
            if count > 0:
                legend_items.append(
                    html.Div([
                        html.Span(
                            label.title(),
                            style={
                                'backgroundColor': color,
                                'color': 'white',
                                'padding': '4px 8px',
                                'borderRadius': '4px',
                                'fontWeight': 'bold',
                                'marginRight': '8px',
                                'fontSize': '12px',
                                'fontFamily': 'monospace'
                            }
                        ),
                        html.Span(f"({count} entities)", style={'fontSize': '12px', 'color': '#6c757d'})
                    ], style={'display': 'inline-block', 'margin': '4px 8px'})
                )
        
        return html.Div(legend_items, style={'textAlign': 'center', 'marginBottom': '20px'})
    
    def _create_dash_app(self):
        """Create the Dash application"""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Entity Ranking Analysis Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': '20px',
                              'color': self.colors['dark'], 'fontFamily': 'Arial, sans-serif'}),
                html.Hr(),
                
                # Summary Stats
                html.Div([
                    html.Div([
                        html.H3(f"{len(self.updated_entities)}", style={'textAlign': 'center'}),
                        html.P("Total Entities", style={'textAlign': 'center', 'color': '#6c757d'})
                    ], style={'width': '25%', 'display': 'inline-block'}),
                    html.Div([
                        html.H3(f"{len(set(e['label'] for e in self.processed_entities))}", style={'textAlign': 'center'}),
                        html.P("Entity Types", style={'textAlign': 'center', 'color': '#6c757d'})
                    ], style={'width': '25%', 'display': 'inline-block'}),
                    html.Div([
                        html.H3(f"{self.auto_query[:20]}..." if len(self.auto_query) > 20 else self.auto_query, style={'textAlign': 'center'}),
                        html.P("Auto Query", style={'textAlign': 'center', 'color': '#6c757d'})
                    ], style={'width': '25%', 'display': 'inline-block'}),
                    html.Div([
                        html.H3(f"{len(self.problem_stmt):,}", style={'textAlign': 'center'}),
                        html.P("Text Length", style={'textAlign': 'center', 'color': '#6c757d'})
                    ], style={'width': '25%', 'display': 'inline-block'}),
                ], style={'backgroundColor': self.colors['light'], 'padding': '20px', 
                         'borderRadius': '5px', 'marginBottom': '20px'})
            ]),
            
            # Navigation Buttons
            html.Div([
                html.Button("Ranking Chart", id="btn-ranking", n_clicks=0, 
                           style={'margin': '5px', 'padding': '10px 20px', 'backgroundColor': self.colors['primary'], 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Button("Distribution", id="btn-distribution", n_clicks=0, 
                           style={'margin': '5px', 'padding': '10px 20px', 'backgroundColor': self.colors['secondary'], 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Button("Scatter Plot", id="btn-scatter", n_clicks=0, 
                           style={'margin': '5px', 'padding': '10px 20px', 'backgroundColor': self.colors['info'], 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Button("Weight System", id="btn-weights", n_clicks=0, 
                           style={'margin': '5px', 'padding': '10px 20px', 'backgroundColor': self.colors['success'], 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Button("Process Flow", id="btn-flow", n_clicks=0, 
                           style={'margin': '5px', 'padding': '10px 20px', 'backgroundColor': self.colors['danger'], 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Button("Metrics Heatmap", id="btn-heatmap", n_clicks=0, 
                           style={'margin': '5px', 'padding': '10px 20px', 'backgroundColor': '#6f42c1', 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Button("Data Table", id="btn-table", n_clicks=0, 
                           style={'margin': '5px', 'padding': '10px 20px', 'backgroundColor': '#20c997', 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Button("Highlighted Text", id="btn-text", n_clicks=0, 
                           style={'margin': '5px', 'padding': '10px 20px', 'backgroundColor': self.colors['warning'], 
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontWeight': 'bold'}),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            # Content Area
            html.Div(id="content-area", children=[
                dcc.Graph(figure=self.create_enhanced_ranking_chart())
            ])
        ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})
        
        # Callbacks
        @app.callback(
            Output('content-area', 'children'),
            [Input('btn-ranking', 'n_clicks'),
             Input('btn-distribution', 'n_clicks'),
             Input('btn-scatter', 'n_clicks'),
             Input('btn-weights', 'n_clicks'),
             Input('btn-flow', 'n_clicks'),
             Input('btn-heatmap', 'n_clicks'),
             Input('btn-table', 'n_clicks'),
             Input('btn-text', 'n_clicks')]
        )
        def update_content(btn_ranking, btn_distribution, btn_scatter, btn_weights, 
                          btn_flow, btn_heatmap, btn_table, btn_text):
            ctx = dash.callback_context
            
            # Default to ranking view
            if not ctx.triggered:
                return [dcc.Graph(figure=self.create_enhanced_ranking_chart())]
            
            # Get the button that was clicked
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Handle each button click
            if button_id == 'btn-ranking':
                return [
                    html.H3("Entity Ranking Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
                    dcc.Graph(figure=self.create_enhanced_ranking_chart(top_n=20)),
                    html.Hr(style={'margin': '20px 0'}),
                    html.H4("Confidence Score Distribution", style={'textAlign': 'center', 'marginBottom': '15px'}),
                    dcc.Graph(figure=self.create_confidence_distribution_chart())
                ]
                
            elif button_id == 'btn-distribution':
                return [
                    html.H3("Entity Type Distribution", style={'textAlign': 'center', 'marginBottom': '20px'}),
                    dcc.Graph(figure=self.create_label_distribution_chart()),
                    html.Div([
                        html.P(f"Total entities: {len(self.processed_entities)}", 
                               style={'textAlign': 'center', 'fontSize': '16px', 'marginTop': '15px'}),
                        html.P(f"Unique entity types: {len(set(e['label'] for e in self.processed_entities))}", 
                               style={'textAlign': 'center', 'fontSize': '16px'})
                    ])
                ]
                
            elif button_id == 'btn-scatter':
                return [
                    html.H3("Position vs Confidence Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
                    dcc.Graph(figure=self.create_metrics_scatter_plot()),
                    html.Div([
                        html.P("This plot shows how entity confidence correlates with position in the text. " +
                               "Higher ranked entities may appear throughout the document.", 
                               style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '15px', 'fontStyle': 'italic'})
                    ])
                ]
            
            elif button_id == 'btn-weights':
                return [
                    html.H3("Entity Ranking Weight System", style={'textAlign': 'center', 'marginBottom': '20px'}),
                    dcc.Graph(figure=self.create_weighting_system_chart()),
                    html.Div([
                        html.P("This visualization shows how different features and entity types are weighted " +
                               "in the ranking algorithm. Higher weights indicate greater importance in the final ranking.", 
                               style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '15px', 'fontStyle': 'italic'})
                    ])
                ]
            
            elif button_id == 'btn-flow':
                return [
                    html.H3("Entity Extraction Process Flow", style={'textAlign': 'center', 'marginBottom': '20px'}),
                    dcc.Graph(figure=self.create_extraction_flow_chart()),
                    html.Div([
                        html.P("This diagram illustrates the hybrid extraction approach combining regex patterns " +
                               "and GLiNER model outputs, followed by feature extraction and final ranking.", 
                               style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '15px', 'fontStyle': 'italic'})
                    ])
                ]
            
            elif button_id == 'btn-heatmap':
                return [
                    html.H3("Entity Metrics Heatmap", style={'textAlign': 'center', 'marginBottom': '20px'}),
                    dcc.Graph(figure=self.create_entity_metrics_heatmap()),
                    html.Div([
                        html.P("This heatmap shows normalized values of key metrics for the top entities. " +
                               "Darker colors indicate higher normalized values within each metric.", 
                               style={'textAlign': 'center', 'fontSize': '14px', 'marginTop': '15px', 'fontStyle': 'italic'})
                    ])
                ]
                
            elif button_id == 'btn-table':
                table_data = self.create_detailed_table()
                if table_data:
                    return [
                        html.H3("Detailed Entity Data", style={'textAlign': 'center', 'marginBottom': '20px'}),
                        html.P(f"Showing all {len(table_data)} entities with detailed metrics", 
                               style={'textAlign': 'center', 'marginBottom': '15px'}),
                        dash_table.DataTable(
                            data=table_data,
                            columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left', 
                                'padding': '10px',
                                'fontSize': '12px',
                                'fontFamily': 'Arial, sans-serif'
                            },
                            style_header={
                                'backgroundColor': self.colors['primary'], 
                                'color': 'white', 
                                'fontWeight': 'bold',
                                'fontSize': '13px'
                            },
                            style_data={
                                'backgroundColor': self.colors['light'],
                                'border': '1px solid #dee2e6'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'white'
                                }
                            ],
                            page_size=25,
                            sort_action="native",
                            filter_action="native"
                        )
                    ]
                else:
                    return [html.P("No data available", style={'textAlign': 'center', 'fontSize': '16px'})]
                    
            elif button_id == 'btn-text':
                top_entities = self.processed_entities[:30]
                highlighted_html = self._render_highlighted_text_html(top_entities)
    
                return [
                    html.H3("Highlighted Problem Statement", 
                        style={'marginBottom': '15px', 'textAlign': 'center', 'color': '#2c3e50'}),

                    # Enhanced description
                    html.Div([
                        html.P([
                            "📊 ", html.Strong("Entity Highlighting Dashboard"), " - ",
                            "Entities ranked by confidence score with visual emphasis on top performers"
                        ], style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '10px', 'color': '#34495e'}),
                        html.P([
                            "🎯 Top 5 entities have ", html.Strong("enhanced borders & shadows"), " | ",
                            "🔢 Rank badges show position | ",
                            "🎨 Opacity fades with ranking"
                        ], style={'textAlign': 'center', 'fontStyle': 'italic', 'marginBottom': '20px', 'color': '#7f8c8d'})
                    ]),

                    # Enhanced statistics bar
                    html.Div([
                        html.Div([
                            html.H4(f"{len(self.processed_entities)}", style={'margin': '0', 'color': '#3498db'}),
                            html.P("Total Entities", style={'margin': '0', 'fontSize': '12px', 'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'flex': '1'}),
                        html.Div([
                            html.H4("30", style={'margin': '0', 'color': '#e74c3c'}),
                            html.P("Shown Below", style={'margin': '0', 'fontSize': '12px', 'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'flex': '1'}),
                        html.Div([
                            html.H4(f"{len(set(e['label'] for e in self.processed_entities))}", style={'margin': '0', 'color': '#2ecc71'}),
                            html.P("Entity Types", style={'margin': '0', 'fontSize': '12px', 'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'flex': '1'}),
                        html.Div([
                            html.H4(f"{max(e['confidence'] for e in self.processed_entities):.3f}", style={'margin': '0', 'color': '#f39c12'}),
                            html.P("Max Confidence", style={'margin': '0', 'fontSize': '12px', 'color': '#7f8c8d'})
                        ], style={'textAlign': 'center', 'flex': '1'})
                    ], style={
                        'display': 'flex', 
                        'backgroundColor': '#ecf0f1', 
                        'padding': '15px', 
                        'borderRadius': '8px', 
                        'marginBottom': '20px',
                        'border': '1px solid #bdc3c7'
                    }),

                    # Enhanced legend with counts
                    html.Div([
                        html.P("🏷️ Entity Type Legend:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'textAlign': 'center'}),
                        html.Div([
                            html.Span(
                                f"{label.title()} ({len([e for e in self.processed_entities if e['label'] == label])})", 
                                style={
                                    'backgroundColor': color,
                                    'color': 'white',
                                    'padding': '6px 12px',
                                    'margin': '3px 6px',
                                    'borderRadius': '20px',
                                    'display': 'inline-block',
                                    'fontSize': '12px',
                                    'fontWeight': 'bold',
                                    'boxShadow': '0 2px 4px rgba(0,0,0,0.2)'
                                }
                            )
                            for label, color in self.entity_colors.items() 
                            if len([e for e in self.processed_entities if e['label'] == label]) > 0
                        ], style={'textAlign': 'center', 'marginBottom': '20px'})
                    ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '8px', 'border': '1px solid #dee2e6'}),

                    # Enhanced ranking guide
                    html.Div([
                        html.P("📋 Visual Guide:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                        html.Div([
                            html.Div([
                                html.Span("🥇 Rank 1-5", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                                html.Span(" - Enhanced borders + shadows", style={'fontSize': '13px', 'color': '#7f8c8d'})
                            ], style={'marginBottom': '5px'}),
                            html.Div([
                                html.Span("🥈 Rank 6-10", style={'fontWeight': 'bold', 'color': '#f39c12'}),
                                html.Span(" - Subtle borders", style={'fontSize': '13px', 'color': '#7f8c8d'})
                            ], style={'marginBottom': '5px'}),
                            html.Div([
                                html.Span("🥉 Rank 11+", style={'fontWeight': 'bold', 'color': '#95a5a6'}),
                                html.Span(" - Standard highlighting with fading opacity", style={'fontSize': '13px', 'color': '#7f8c8d'})
                            ], style={'marginBottom': '5px'}),
                            html.Div([
                                html.Span("💡 Hover", style={'fontWeight': 'bold', 'color': '#3498db'}),
                                html.Span(" - View detailed entity information", style={'fontSize': '13px', 'color': '#7f8c8d'})
                            ])
                        ], style={'fontSize': '14px'})
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px', 'marginBottom': '20px', 'border': '1px solid #e9ecef'}),

                    # The highlighted text component
                    html.Div([
                        html.H4("📄 Problem Statement with Highlighted Entities", 
                            style={'marginBottom': '15px', 'color': '#2c3e50'}),
                        html.Div([
                            dcc.Markdown(
                                children=highlighted_html,
                                dangerously_allow_html=True,
                                style={
                                    'fontFamily': 'monospace',
                                    'whiteSpace': 'pre-wrap',
                                    'lineHeight': '1.5'
                                }
                            )
                        ], style={
                            'backgroundColor': 'white',
                            'padding': '20px',
                            'borderRadius': '8px',
                            'border': '1px solid #dee2e6',
                            'maxHeight': '600px',
                            'overflowY': 'auto'
                        })
                    ])
                ]
            
            # Default fallback (moved outside and properly indented)
            return [dcc.Graph(figure=self.create_enhanced_ranking_chart())]
        
        return app
    
    def run_dashboard(self, debug=False, port=8050):
        """Run the dashboard on localhost"""
        def open_browser():
            time.sleep(1.5)
            try:
                webbrowser.open_new(f'http://localhost:{port}/')
            except Exception as e:
                print(f"Could not open browser automatically: {e}")
                print(f"Please manually open: http://localhost:{port}/")
        
        if not debug:
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
        
        print(f"🚀 Starting Entity Ranking Dashboard...")
        print(f"📊 Dashboard will be available at: http://localhost:{port}/")
        print(f"📈 Total entities analyzed: {len(self.updated_entities)}")
        print(f"🔍 Auto-generated query: '{self.auto_query}'")
        print(f"⚡ Ready to explore your entity rankings!")
        
        try:
            # Use app.run with explicit parameters
            self.app.run(debug=debug, port=port, host='127.0.0.1', use_reloader=False)
        except Exception as e:
            print(f"Error starting dashboard: {e}")
            print("Try running from command line instead of VS Code debugger")

def main():
    """Main function to run the visualization dashboard"""
    print("="*80)
    print("PROFESSIONAL ENTITY RANKING VISUALIZATION DASHBOARD")
    print("="*80)
    print(f"Repository: {repo_name}")
    print(f"Version: {version}")
    print(f"Date: {date}")
    print(f"Commit: {commit}")
    print("="*80)
    
    try:
        # Create and run the visualizer
        visualizer = ProfessionalEntityVisualizer()
        visualizer.run_dashboard(debug=False, port=8050)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("💡 Try running from terminal instead of VS Code debugger")

if __name__ == "__main__":
    main()