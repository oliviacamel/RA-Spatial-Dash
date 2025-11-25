import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import colorcet as cc
import pickle
import os
import pandas as pd

app = dash.Dash(__name__)
server = app.server
# Load the parquet file
PARQUET_FILE = 'dataParquet/fRARCx3ERT2_all/'
CELLTYPE_FILE = 'data/leiden_membership.npy'
CELLTYPE_FILE = np.load(CELLTYPE_FILE)
CELLTYPECOLORS = cc.glasbey[:max(np.unique(CELLTYPE_FILE))+1]
with open ('data/gene_marker.pkl','rb') as f :
    gene_markers = pickle.load(f)
if os.path.exists(PARQUET_FILE):
    df = pd.read_parquet(f'{PARQUET_FILE}')
    #print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    #df = pd.read_csv(f'{PARQUET_FILE}',index_col = 0)
    #print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    x_col = 'x'
    y_col = 'y'

    # Get all numeric columns for the dropdown (excluding x and y)
    feature_cols = df.columns[6:].tolist()
    #print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")

# App layout
app.layout = html.Div([
    html.H1("fRARCx3ERT2_all Tissue Visualization",
            style={'textAlign': 'center', 'marginTop': '20px'}),

    html.Div([
        # Feature Visualization
        html.Div([
            html.H3("Feature Visualization", style={'textAlign': 'center', 'marginTop': '20px'}),
            html.Div([
                html.Label("Select Feature to Color By:",
                           style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in feature_cols],
                    value=feature_cols[0] if feature_cols else None,
                    style={'width': '300px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center',
                      'justifyContent': 'center', 'marginBottom': '20px'}),
            dcc.Graph(id='feature-scatter-plot', style={'height': '70vh'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

        # Cluster Visualization
        html.Div([
            html.H3("Cluster Visualization", style={'textAlign': 'center', 'marginTop': '20px'}),
            html.Div([
                html.Label("Select cell type to Color By:",
                           style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='cluster-dropdown',
                    options=[{'label': f"cell cluster {i}", 'value': i} for i in np.unique(CELLTYPE_FILE)],
                    value=[np.unique(CELLTYPE_FILE)[0]] if len(np.unique(CELLTYPE_FILE)) > 0 else [],
                    multi=True,
                    style={'width': '300px'}
                )
            ], style={'display': 'flex', 'alignItems': 'center',
                      'justifyContent': 'center', 'marginBottom': '20px'}),
            dcc.Graph(id='cluster-scatter-plot', style={'height': '70vh'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    # Bottom row: Heatmap subplots
    html.Div([
        html.H3("Top Features per Selected Cluster", style={'textAlign': 'center', 'marginTop': '40px'}),
        #html.P("Heatmaps automatically generated for selected clusters",
        #       style={'textAlign': 'center', 'color': 'gray'}),
        dcc.Graph(id='cluster-heatmap', style={'height': '60vh'})
    ], style={'marginTop': '40px', 'padding': '20px'})

], style={'fontFamily': 'Arial, sans-serif'})

# Callback to update the feature scatter plot
@app.callback(
    Output('feature-scatter-plot', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_scatter(selected_feature):
    if selected_feature is None:
        return go.Figure()

    df_copy = df[[x_col, y_col, selected_feature]].copy()
    df_copy[selected_feature] = np.log2(df_copy[selected_feature] + 1e-10)  # Add small value to avoid log(0)
    df_copy = df_copy[(~np.isinf(df_copy[selected_feature]))&(df_copy[selected_feature]>0)]

    # Create a new figure for this callback
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=df_copy[x_col],
            y=df_copy[y_col],
            mode='markers',
            marker=dict(
                size=1,
                color=df_copy[selected_feature],
                colorscale='Reds',
                colorbar=dict(title=f'log({selected_feature})'),
                cmin=np.quantile(df_copy[selected_feature], 0.1),
                cmax=np.quantile(df_copy[selected_feature], 1),
                opacity=0.8
            ),
            name=selected_feature,
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<br><b>{selected_feature}</b>: %{{marker.color:.2f}}<extra></extra>'
        )
    )

    fig.update_layout(
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig

# Callback to update the cluster scatter plot
@app.callback(
    Output('cluster-scatter-plot', 'figure'),
    Input('cluster-dropdown', 'value')
)
def update_celltype_scatter(selected_clusters):
    if selected_clusters is None or len(selected_clusters) == 0:
        return go.Figure()

    # Create a new figure for this callback
    fig = go.Figure()

    # Plot each cell type separately for better legend control
    for c, celltype in enumerate(selected_clusters):
        df_subset = df[CELLTYPE_FILE == celltype].copy()
        color = CELLTYPECOLORS[celltype]
        fig.add_trace(
            go.Scattergl(
                x=df_subset[x_col],
                y=df_subset[y_col],
                mode='markers',
                marker=dict(
                    size=1.5,
                    opacity=0.9,
                    color=color
                ),
                name=f'cluster {celltype}',
                hovertemplate=f'<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<br><b>Cell Type</b>: {celltype}<extra></extra>'
            )
        )

    fig.update_layout(
        height=800,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

# Callback to update the heatmap with subplots
@app.callback(
    Output('cluster-heatmap', 'figure'),
    Input('cluster-dropdown', 'value')
)
def update_heatmap(selected_clusters):
    if selected_clusters is None or len(selected_clusters) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Select clusters to view their top features",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    n_clusters = len(selected_clusters)

    fig = make_subplots(
        rows=2,
        cols=n_clusters,
        subplot_titles=[f'Cluster {c}' for c in selected_clusters],
        horizontal_spacing=0.15 / n_clusters if n_clusters > 1 else 0.1,
        vertical_spacing= 0.2,
        specs=[[{'type': 'heatmap'} for _ in range(n_clusters)] for _ in range(2)]
    )

    # For each selected cluster, create a heatmap
    for idx, cluster_id in enumerate(selected_clusters):
        # Get top features for this cluster from dictionary
        # Get data for both heatmaps
        data_in_cluster = df[CELLTYPE_FILE == cluster_id].loc[:, gene_markers[cluster_id]['gene']]
        data_out_cluster = df[CELLTYPE_FILE != cluster_id].loc[:, gene_markers[cluster_id]['gene']]

        # Calculate global min/max for this cluster across both datasets
        combined_data = np.concatenate([data_in_cluster.values.flatten(),
                                      data_out_cluster.values.flatten()])
        vmin = np.percentile(combined_data,20)
        vmax = np.percentile(combined_data,80)
        # Add heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=df[CELLTYPE_FILE==cluster_id].loc[:,gene_markers[cluster_id]['gene']],
                x=gene_markers[cluster_id]['gene'],
                #y=[f'Cluster {cluster_id}'],
                colorscale='Reds',
                zmin= vmin,
                zmax= vmax,

                colorbar=dict(
                    len=0.1,
                    y=-0.15,
                    yanchor='middle',
                    x=0.2 + (idx * 0.8/n_clusters),
                    xanchor='left',
                    orientation = 'h',
                    thickness=10
                ),
                hovertemplate='<b>Feature</b>: %{x}<br><b>Expression</b>: %{z:.2f}<extra></extra>',
                showscale=False
            ),
            row=1,
            col=idx+1
        )
        fig.add_trace(
            go.Heatmap(
                z=df[CELLTYPE_FILE!=cluster_id].loc[:,gene_markers[cluster_id]['gene']],
                x=gene_markers[cluster_id]['gene'],
                #y=[f'Cluster {cluster_id}'],
                colorscale='Reds',
                zmin= vmin,
                zmax= vmax,
                colorbar=dict(
                    len=0.1,
                    y=-0.15,
                    yanchor='middle',
                    x=0.2 + (idx * 0.8/n_clusters),
                    orientation = 'h',
                    xanchor='left',
                    thickness=10
                ),
                hovertemplate='<b>Feature</b>: %{x}<br><b>Expression</b>: %{z:.2f}<extra></extra>',
                showscale=False
            ),
            row=2,
            col=idx+1
        )

        # Update x-axis for this subplot
        fig.update_xaxes(
            tickangle=0,
            row=1,
            col=idx + 1
        )

        # Update y-axis
        fig.update_xaxes(
            showticklabels=True,
            tickangle=-45,
            row=2,
            col=idx+1
        )
        fig.update_yaxes(
        title_text="In Cluster",
        showticklabels=True,
        row=1,
        col=idx+1,
        title_standoff=10
    )
    fig.update_yaxes(
        title_text="Out Cluster",
        showticklabels=True,
        row=2,
        col=idx+1,
        title_standoff=10
    )

    # Update overall layout
    fig.update_layout(
        height=max(400, 200 * n_clusters),  # Dynamic height based on number of clusters
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,1)',
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=120)
    )
    return fig

if __name__ == '__main__':
    #print(f"Memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
    app.run()

    
