#!/usr/bin/env python3
"""
chart_generator.py

Generate various types of charts and visualizations from data.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartGenerator:
    def __init__(self):
        self.supported_formats = ['png', 'jpg', 'jpeg', 'pdf', 'svg', 'html']
        self.chart_types = [
            'bar', 'line', 'scatter', 'histogram', 'box', 'violin',
            'heatmap', 'pie', 'area', 'bubble', 'correlation', 'distribution'
        ]
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        path = Path(data_path)
        
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        elif path.suffix.lower() == '.json':
            df = pd.read_json(path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        elif path.suffix.lower() == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix.lower() == '.tsv':
            df = pd.read_csv(path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return df
    
    def auto_detect_chart_type(self, df: pd.DataFrame, x_col: str = None, 
                              y_col: str = None) -> str:
        """Automatically detect the best chart type for the data"""
        
        if x_col and y_col:
            x_series = df[x_col]
            y_series = df[y_col]
            
            # Both numeric -> scatter plot
            if pd.api.types.is_numeric_dtype(x_series) and pd.api.types.is_numeric_dtype(y_series):
                return 'scatter'
            
            # X categorical, Y numeric -> bar chart
            elif not pd.api.types.is_numeric_dtype(x_series) and pd.api.types.is_numeric_dtype(y_series):
                return 'bar'
            
            # X numeric, Y categorical -> horizontal bar
            elif pd.api.types.is_numeric_dtype(x_series) and not pd.api.types.is_numeric_dtype(y_series):
                return 'bar'
            
            # Both categorical -> heatmap
            else:
                return 'heatmap'
        
        elif x_col:
            x_series = df[x_col]
            
            # Single numeric column -> histogram
            if pd.api.types.is_numeric_dtype(x_series):
                return 'histogram'
            
            # Single categorical column -> pie chart
            else:
                return 'pie'
        
        # Multiple numeric columns -> correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            return 'correlation'
        
        # Default to bar chart
        return 'bar'
    
    def create_matplotlib_chart(self, df: pd.DataFrame, chart_type: str, 
                               x_col: str = None, y_col: str = None,
                               title: str = None, **kwargs) -> Any:
        """Create chart using matplotlib/seaborn"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure
            fig_size = kwargs.get('figsize', (10, 6))
            fig, ax = plt.subplots(figsize=fig_size)
            
            if chart_type == 'bar':
                if x_col and y_col:
                    df_plot = df.groupby(x_col)[y_col].mean().reset_index()
                    ax.bar(df_plot[x_col], df_plot[y_col])
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(f'Average {y_col}')
                else:
                    # Value counts of first categorical column
                    cat_col = df.select_dtypes(include=['object']).columns[0]
                    value_counts = df[cat_col].value_counts()
                    ax.bar(value_counts.index, value_counts.values)
                    ax.set_xlabel(cat_col)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
            
            elif chart_type == 'line':
                if x_col and y_col:
                    ax.plot(df[x_col], df[y_col], marker='o')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                else:
                    # Plot first numeric column over index
                    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                    ax.plot(df.index, df[numeric_col], marker='o')
                    ax.set_xlabel('Index')
                    ax.set_ylabel(numeric_col)
            
            elif chart_type == 'scatter':
                if x_col and y_col:
                    ax.scatter(df[x_col], df[y_col], alpha=0.6)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                else:
                    # Scatter plot of first two numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
                    if len(numeric_cols) >= 2:
                        ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                        ax.set_xlabel(numeric_cols[0])
                        ax.set_ylabel(numeric_cols[1])
            
            elif chart_type == 'histogram':
                if x_col:
                    ax.hist(df[x_col].dropna(), bins=kwargs.get('bins', 30), alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel('Frequency')
                else:
                    # Histogram of first numeric column
                    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                    ax.hist(df[numeric_col].dropna(), bins=kwargs.get('bins', 30), alpha=0.7)
                    ax.set_xlabel(numeric_col)
                    ax.set_ylabel('Frequency')
            
            elif chart_type == 'box':
                if x_col and y_col:
                    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                else:
                    # Box plot of numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
                    df[numeric_cols].boxplot(ax=ax)
                    plt.xticks(rotation=45)
            
            elif chart_type == 'violin':
                if x_col and y_col:
                    sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
                    plt.xticks(rotation=45)
                else:
                    logger.warning("Violin plot requires both x and y columns")
            
            elif chart_type == 'heatmap':
                if x_col and y_col:
                    # Cross-tabulation heatmap
                    crosstab = pd.crosstab(df[x_col], df[y_col])
                    sns.heatmap(crosstab, annot=True, fmt='d', ax=ax)
                else:
                    # Correlation heatmap
                    numeric_df = df.select_dtypes(include=[np.number])
                    if numeric_df.shape[1] > 1:
                        corr = numeric_df.corr()
                        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                    else:
                        logger.warning("Not enough numeric columns for correlation heatmap")
            
            elif chart_type == 'pie':
                if x_col:
                    value_counts = df[x_col].value_counts()
                else:
                    # Pie chart of first categorical column
                    cat_col = df.select_dtypes(include=['object']).columns[0]
                    value_counts = df[cat_col].value_counts()
                
                # Limit to top categories
                if len(value_counts) > 10:
                    top_values = value_counts.head(10)
                    other_sum = value_counts.tail(len(value_counts) - 10).sum()
                    if other_sum > 0:
                        top_values['Other'] = other_sum
                    value_counts = top_values
                
                ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            
            elif chart_type == 'area':
                if x_col and y_col:
                    ax.fill_between(df[x_col], df[y_col], alpha=0.7)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                else:
                    # Area plot of first numeric column
                    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                    ax.fill_between(df.index, df[numeric_col], alpha=0.7)
                    ax.set_xlabel('Index')
                    ax.set_ylabel(numeric_col)
            
            elif chart_type == 'correlation':
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] > 1:
                    corr = numeric_df.corr()
                    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                else:
                    logger.warning("Not enough numeric columns for correlation matrix")
            
            # Set title
            if title:
                ax.set_title(title)
            elif x_col and y_col:
                ax.set_title(f'{chart_type.title()} Chart: {x_col} vs {y_col}')
            elif x_col:
                ax.set_title(f'{chart_type.title()} Chart: {x_col}')
            else:
                ax.set_title(f'{chart_type.title()} Chart')
            
            plt.tight_layout()
            return fig
        
        except ImportError:
            logger.error("matplotlib/seaborn not available for chart generation")
            return None
    
    def create_plotly_chart(self, df: pd.DataFrame, chart_type: str,
                           x_col: str = None, y_col: str = None,
                           title: str = None, **kwargs) -> Any:
        """Create interactive chart using plotly"""
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            
            if chart_type == 'bar':
                if x_col and y_col:
                    df_plot = df.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(df_plot, x=x_col, y=y_col)
                else:
                    # Value counts of first categorical column
                    cat_col = df.select_dtypes(include=['object']).columns[0]
                    value_counts = df[cat_col].value_counts().reset_index()
                    value_counts.columns = [cat_col, 'count']
                    fig = px.bar(value_counts, x=cat_col, y='count')
            
            elif chart_type == 'line':
                if x_col and y_col:
                    fig = px.line(df, x=x_col, y=y_col, markers=True)
                else:
                    # Line plot of first numeric column
                    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                    fig = px.line(df, x=df.index, y=numeric_col, markers=True)
            
            elif chart_type == 'scatter':
                if x_col and y_col:
                    fig = px.scatter(df, x=x_col, y=y_col)
                else:
                    # Scatter plot of first two numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:2]
                    if len(numeric_cols) >= 2:
                        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            
            elif chart_type == 'histogram':
                if x_col:
                    fig = px.histogram(df, x=x_col, nbins=kwargs.get('bins', 30))
                else:
                    # Histogram of first numeric column
                    numeric_col = df.select_dtypes(include=[np.number]).columns[0]
                    fig = px.histogram(df, x=numeric_col, nbins=kwargs.get('bins', 30))
            
            elif chart_type == 'box':
                if x_col and y_col:
                    fig = px.box(df, x=x_col, y=y_col)
                else:
                    # Box plot of numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
                    fig = px.box(df[numeric_cols])
            
            elif chart_type == 'violin':
                if x_col and y_col:
                    fig = px.violin(df, x=x_col, y=y_col)
                else:
                    logger.warning("Violin plot requires both x and y columns")
                    return None
            
            elif chart_type == 'heatmap':
                if x_col and y_col:
                    # Cross-tabulation heatmap
                    crosstab = pd.crosstab(df[x_col], df[y_col])
                    fig = px.imshow(crosstab, text_auto=True)
                else:
                    # Correlation heatmap
                    numeric_df = df.select_dtypes(include=[np.number])
                    if numeric_df.shape[1] > 1:
                        corr = numeric_df.corr()
                        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu')
            
            elif chart_type == 'pie':
                if x_col:
                    value_counts = df[x_col].value_counts()
                else:
                    # Pie chart of first categorical column
                    cat_col = df.select_dtypes(include=['object']).columns[0]
                    value_counts = df[cat_col].value_counts()
                
                # Limit to top categories
                if len(value_counts) > 10:
                    top_values = value_counts.head(10)
                    other_sum = value_counts.tail(len(value_counts) - 10).sum()
                    if other_sum > 0:
                        top_values['Other'] = other_sum
                    value_counts = top_values
                
                fig = px.pie(values=value_counts.values, names=value_counts.index)
            
            elif chart_type == 'bubble':
                if len(df.select_dtypes(include=[np.number]).columns) >= 3:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], 
                                   size=numeric_cols[2])
                else:
                    logger.warning("Bubble chart requires at least 3 numeric columns")
                    return None
            
            else:
                logger.warning(f"Plotly chart type {chart_type} not implemented")
                return None
            
            # Set title
            if title:
                fig.update_layout(title=title)
            elif x_col and y_col:
                fig.update_layout(title=f'{chart_type.title()} Chart: {x_col} vs {y_col}')
            elif x_col:
                fig.update_layout(title=f'{chart_type.title()} Chart: {x_col}')
            else:
                fig.update_layout(title=f'{chart_type.title()} Chart')
            
            return fig
        
        except ImportError:
            logger.error("plotly not available for interactive charts")
            return None
    
    def save_chart(self, fig: Any, output_path: str, chart_format: str = 'png'):
        """Save chart to file"""
        
        if fig is None:
            logger.error("No figure to save")
            return
        
        try:
            # Determine if it's matplotlib or plotly figure
            if hasattr(fig, 'savefig'):  # matplotlib
                fig.savefig(output_path, format=chart_format, dpi=300, bbox_inches='tight')
                logger.info(f"Chart saved to {output_path}")
            
            elif hasattr(fig, 'write_image') or hasattr(fig, 'write_html'):  # plotly
                if chart_format.lower() == 'html':
                    fig.write_html(output_path)
                else:
                    fig.write_image(output_path, format=chart_format)
                logger.info(f"Interactive chart saved to {output_path}")
            
            else:
                logger.error("Unknown figure type for saving")
        
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
    
    def create_dashboard(self, df: pd.DataFrame, output_path: str):
        """Create a simple dashboard with multiple charts"""
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Data Overview', 'Distribution', 'Correlation', 'Categories'],
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'scatter'}, {'type': 'xy'}]]
            )
            
            # Chart 1: Basic stats table (using scatter for text display)
            stats_text = f"Rows: {len(df)}<br>Columns: {len(df.columns)}<br>Memory: {df.memory_usage(deep=True).sum()/1024/1024:.1f} MB"
            fig.add_trace(
                go.Scatter(x=[0], y=[0], mode='text', text=[stats_text], textfont_size=14),
                row=1, col=1
            )
            
            # Chart 2: Distribution of first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig.add_trace(
                    go.Histogram(x=df[numeric_cols[0]], name=numeric_cols[0]),
                    row=1, col=2
                )
            
            # Chart 3: Correlation heatmap (simplified)
            if len(numeric_cols) > 1:
                corr = df[numeric_cols[:5]].corr()  # Limit to 5 columns
                fig.add_trace(
                    go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, 
                             colorscale='RdBu', zmid=0),
                    row=2, col=1
                )
            
            # Chart 4: Category distribution
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                value_counts = df[cat_cols[0]].value_counts().head(10)
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, name=cat_cols[0]),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Data Dashboard",
                height=800,
                showlegend=False
            )
            
            # Save dashboard
            fig.write_html(output_path)
            logger.info(f"Dashboard saved to {output_path}")
            
            return fig
        
        except ImportError:
            logger.error("plotly not available for dashboard creation")
            return None

def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample data for testing"""
    np.random.seed(42)
    
    data = {
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
        'value1': np.random.normal(100, 15, size=n_samples),
        'value2': np.random.normal(50, 10, size=n_samples),
        'score': np.random.uniform(0, 100, size=n_samples),
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D')[:n_samples]
    }
    
    # Add some correlation
    data['value3'] = data['value1'] * 0.7 + np.random.normal(0, 5, size=n_samples)
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description="Generate charts and visualizations from data")
    parser.add_argument('--data', help='Path to data file')
    parser.add_argument('--chart-type', choices=['auto'] + ChartGenerator().chart_types,
                       default='auto', help='Type of chart to generate')
    parser.add_argument('--x-column', help='X-axis column')
    parser.add_argument('--y-column', help='Y-axis column')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['png', 'jpg', 'pdf', 'svg', 'html'],
                       default='png', help='Output format')
    parser.add_argument('--title', help='Chart title')
    parser.add_argument('--interactive', action='store_true', help='Create interactive chart')
    parser.add_argument('--dashboard', action='store_true', help='Create dashboard')
    parser.add_argument('--sample-data', action='store_true', help='Use sample data')
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 6],
                       help='Figure size (width height)')
    parser.add_argument('--bins', type=int, default=30, help='Number of bins for histogram')
    
    args = parser.parse_args()
    
    generator = ChartGenerator()
    
    # Load or create data
    if args.sample_data:
        print("Creating sample data...")
        df = create_sample_data()
        print(f"Created sample data with {len(df)} rows and {len(df.columns)} columns")
    elif args.data:
        print(f"Loading data from {args.data}...")
        df = generator.load_data(args.data)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    else:
        print("Error: Either --data or --sample-data must be specified")
        return
    
    # Show data info
    print(f"\nDataset Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Data types: {df.dtypes.value_counts().to_dict()}")
    
    # Auto-detect chart type if needed
    chart_type = args.chart_type
    if chart_type == 'auto':
        chart_type = generator.auto_detect_chart_type(df, args.x_column, args.y_column)
        print(f"  Auto-detected chart type: {chart_type}")
    
    # Set output file
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"chart_{chart_type}_{timestamp}.{args.format}"
    
    # Create dashboard or single chart
    if args.dashboard:
        if not output_file.endswith('.html'):
            output_file = output_file.rsplit('.', 1)[0] + '.html'
        
        print(f"Creating dashboard...")
        fig = generator.create_dashboard(df, output_file)
    
    else:
        # Create chart
        print(f"Creating {chart_type} chart...")
        
        if args.interactive or args.format == 'html':
            fig = generator.create_plotly_chart(
                df, chart_type, args.x_column, args.y_column, args.title,
                figsize=args.figsize, bins=args.bins
            )
        else:
            fig = generator.create_matplotlib_chart(
                df, chart_type, args.x_column, args.y_column, args.title,
                figsize=args.figsize, bins=args.bins
            )
        
        if fig:
            generator.save_chart(fig, output_file, args.format)
        else:
            print("Failed to create chart")
            return
    
    print(f"Chart saved to: {output_file}")
    
    # Show chart info
    if args.x_column and args.y_column:
        print(f"Chart: {chart_type} of {args.x_column} vs {args.y_column}")
    elif args.x_column:
        print(f"Chart: {chart_type} of {args.x_column}")
    else:
        print(f"Chart: {chart_type} of dataset")

if __name__ == "__main__":
    main()