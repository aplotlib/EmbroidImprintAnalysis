import pandas as pd
import streamlit as st
import io
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import gc
import numpy as np
import json
import time
import re

# Configure page
st.set_page_config(
    page_title="Embroid/Imprint Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for modern, professional look
st.markdown("""
<style>
    /* Modern, clean theme */
    .stApp {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Header section */
    .main-header {
        background: #ffffff;
        padding: 2rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    /* Metric cards with gradient borders */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 2px solid #667eea;  /* Fallback solid border */
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    /* Gradient border using pseudo-element (better browser support) */
    [data-testid="metric-container"]::after {
        content: '';
        position: absolute;
        top: -2px; right: -2px; bottom: -2px; left: -2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        z-index: -1;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] > div {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="metric-container"] label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="metric-container"] > div[data-testid="metric-container-value"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.01em;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #e5e7eb;
        border-radius: 12px;
        background: #fafafa;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: #f9fafb;
    }
    
    /* Data tables */
    .dataframe {
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        border: none !important;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: #f3f4f6 !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.05em;
        color: #4b5563;
        padding: 12px !important;
    }
    
    .dataframe tbody tr:hover {
        background: #f9fafb !important;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    .info-card h4 {
        font-size: 1.125rem;
        margin-bottom: 0.5rem;
        color: #374151;
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 0.75rem 0;
        background: none;
        border: none;
        color: #6b7280;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #667eea;
        border-bottom: 2px solid #667eea;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        background: #f9fafb;
        border-radius: 8px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .insight-card strong {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

def parse_hierarchical_sheet(df, sheet_name):
    """Parse the hierarchical format where SKUs contain customer data"""
    customer_totals = {}
    current_sku = None
    rows_processed = 0
    
    # Find the data start row (skip headers)
    data_start_row = 0
    for i in range(min(10, len(df))):
        first_cell = str(df.iloc[i, 0] if pd.notna(df.iloc[i, 0]) else '')
        if 'Total' in first_cell or 'total' in first_cell.lower():
            data_start_row = i + 1
            break
    
    # Process each row
    for i in range(data_start_row, len(df)):
        if i % 1000 == 0:  # Progress indicator for large files
            rows_processed = i
            
        try:
            first_cell = str(df.iloc[i, 0] if pd.notna(df.iloc[i, 0]) else '').strip()
            
            if not first_cell:
                continue
            
            # Get the total from the last column
            last_col_idx = len(df.columns) - 1
            total_value = df.iloc[i, last_col_idx]
            
            # Check if this is a SKU row (contains brackets)
            if '[' in first_cell and ']' in first_cell:
                # Extract SKU code - handle nested or malformed brackets
                try:
                    sku_match = re.search(r'\[([^\]]+)\]', first_cell)
                    if sku_match:
                        current_sku = sku_match.group(1).strip()
                except Exception:
                    current_sku = None
            else:
                # This should be a customer row
                # Remove leading spaces
                customer_name = first_cell.lstrip()
                
                # Skip if it looks like a subtotal or header
                if any(skip in customer_name.lower() for skip in ['total', 'qty delivered', 'sum', 'grand total']):
                    continue
                
                # Only process if we have a valid total
                if pd.notna(total_value) and isinstance(total_value, (int, float)) and total_value > 0:
                    # Clean customer name (remove duplicate contact info)
                    parts = customer_name.split(',')
                    if len(parts) == 2 and parts[0].strip() == parts[1].strip():
                        customer_name = parts[0].strip()
                    elif len(parts) > 1:
                        # Keep first part as company name
                        customer_name = parts[0].strip()
                    
                    # Skip empty customer names
                    if not customer_name or customer_name.lower() in ['', 'total', 'subtotal']:
                        continue
                    
                    # Add to customer totals
                    if customer_name not in customer_totals:
                        customer_totals[customer_name] = {
                            'Customer': customer_name,
                            'Total Quantity': 0,
                            'SKUs': [],
                            'SKU_List': set()
                        }
                    
                    customer_totals[customer_name]['Total Quantity'] += int(total_value)
                    if current_sku:
                        customer_totals[customer_name]['SKU_List'].add(current_sku)
        
        except Exception as e:
            # Skip problematic rows
            continue
    
    # Convert to list format
    customer_list = []
    for customer, data in customer_totals.items():
        sku_list = sorted(list(data['SKU_List']))
        data['SKUs'] = ', '.join(sku_list[:5])  # Top 5 SKUs
        data['SKU Count'] = len(sku_list)
        del data['SKU_List']  # Remove the set
        customer_list.append(data)
    
    return customer_list

def process_excel_file(file_content):
    """Main function to process the Excel file"""
    results = {}
    processing_stats = {
        'total_customers': 0,
        'total_quantity': 0,
        'unique_skus': set(),
        'sheets_processed': []
    }
    
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(io.BytesIO(file_content))
        
        # Process each sheet
        for sheet_name in excel_file.sheet_names:
            try:
                with st.spinner(f'Processing sheet: {sheet_name}...'):
                    # Read the sheet
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                    
                    if df.empty or len(df) < 5:
                        continue
                    
                    # Determine category from sheet name
                    sheet_lower = sheet_name.lower()
                    if 'imprint' in sheet_lower:
                        category = 'imprinting'
                    elif 'embroid' in sheet_lower:
                        category = 'embroidery'
                    else:
                        # Skip sheets that don't match our categories
                        continue
                    
                    # Parse the hierarchical data
                    customer_data = parse_hierarchical_sheet(df, sheet_name)
                    
                    if customer_data:
                        # Convert to DataFrame
                        result_df = pd.DataFrame(customer_data)
                        result_df = result_df.sort_values('Total Quantity', ascending=False)
                        result_df = result_df.reset_index(drop=True)
                        
                        # Add rank
                        result_df['Rank'] = range(1, len(result_df) + 1)
                        
                        # Calculate cumulative percentage
                        result_df['Cumulative %'] = (result_df['Total Quantity'].cumsum() / result_df['Total Quantity'].sum() * 100)
                        
                        # Rename columns appropriately
                        if category == 'embroidery':
                            result_df = result_df.rename(columns={'Customer': 'Company'})
                        
                        results[category] = result_df
                        processing_stats['sheets_processed'].append(sheet_name)
                        
                        # Update stats
                        processing_stats['total_customers'] += len(result_df)
                        processing_stats['total_quantity'] += result_df['Total Quantity'].sum()
                        
                        # Collect unique SKUs
                        for sku_list in result_df['SKUs']:
                            if sku_list:
                                processing_stats['unique_skus'].update(sku_list.split(', '))
            
            except Exception as e:
                st.warning(f"Could not process sheet '{sheet_name}': {str(e)}")
                continue
        
        # Calculate average order size
        if processing_stats['total_customers'] > 0:
            processing_stats['avg_order_size'] = round(
                processing_stats['total_quantity'] / processing_stats['total_customers'], 1
            )
        
        return results, processing_stats
    
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")

def create_visualizations(results):
    """Create simplified, insightful visualizations"""
    viz = {}
    
    for category, df in results.items():
        if df is not None and not df.empty:
            # 1. Customer Tier Distribution (Sunburst or Donut)
            # Make a copy to avoid modifying original
            df_copy = df.copy()
            
            # Ensure df is sorted by Total Quantity descending
            if 'Total Quantity' in df_copy.columns:
                df_copy = df_copy.sort_values('Total Quantity', ascending=False)
            
            # Calculate tiers based on 80/20 rule
            total_qty = df_copy['Total Quantity'].sum()
            df_copy['Tier'] = 'C (Standard)'  # Default
            
            cumulative = 0
            for idx, row in df_copy.iterrows():
                cumulative += row['Total Quantity']
                if cumulative <= total_qty * 0.2:  # Top 20% of volume
                    df_copy.at[idx, 'Tier'] = 'A (Strategic)'
                elif cumulative <= total_qty * 0.5:  # Next 30% of volume
                    df_copy.at[idx, 'Tier'] = 'B (Important)'
                else:
                    df_copy.at[idx, 'Tier'] = 'C (Standard)'
            
            # Determine customer column name
            customer_col = 'Company' if 'Company' in df_copy.columns else 'Customer'
            
            # Count customers per tier
            tier_counts = df_copy.groupby('Tier').agg({
                customer_col: 'count',
                'Total Quantity': 'sum'
            }).reset_index()
            tier_counts.columns = ['Tier', 'Customer Count', 'Total Volume']
            
            # Ensure we have tier data before creating chart
            if len(tier_counts) > 0:
                # Create donut chart
                fig_donut = go.Figure(data=[go.Pie(
                    labels=tier_counts['Tier'],
                    values=tier_counts['Total Volume'],
                    hole=.7,
                    marker_colors=['#667eea', '#a78bfa', '#e0e7ff'],
                    textinfo='label+percent',
                    textposition='outside',
                    textfont=dict(size=14, family='Inter, sans-serif')
                )])
                
                fig_donut.update_layout(
                    title=dict(
                        text=f"{category.title()} Customer Segmentation",
                        font=dict(size=18, family='Inter, sans-serif', color='#1a1a1a')
                    ),
                    showlegend=False,
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=60, b=0, l=0, r=0),
                    annotations=[
                        dict(
                            text=f"{len(df)}<br>Customers",
                            x=0.5, y=0.5,
                            font=dict(size=24, family='Inter, sans-serif', color='#1a1a1a', weight=600),
                            showarrow=False
                        )
                    ]
                )
                
                viz[f'{category}_donut'] = fig_donut
            
            # 2. Volume Concentration Curve (simplified Pareto)
            fig_concentration = go.Figure()
            
            # Check if Cumulative % exists and has data
            if 'Cumulative %' in df.columns and len(df) > 0:
                # Add area chart for cumulative percentage
                fig_concentration.add_trace(go.Scatter(
                    x=list(range(1, min(51, len(df) + 1))),  # Top 50 customers
                    y=df['Cumulative %'][:50],
                    mode='lines',
                    name='Volume Concentration',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ))
                
                # Add 80% reference line if we have data
                if len(df) > 0 and 'Cumulative %' in df.columns:
                    fig_concentration.add_hline(
                        y=80, 
                        line_dash="dash", 
                        line_color="#9ca3af",
                        annotation_text="80% Volume",
                        annotation_position="right"
                    )
            
            fig_concentration.update_layout(
                title=dict(
                    text=f"{category.title()} Volume Concentration",
                    font=dict(size=18, family='Inter, sans-serif', color='#1a1a1a')
                ),
                xaxis_title="Customer Rank",
                yaxis_title="Cumulative Volume %",
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#f3f4f6',
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='#f3f4f6',
                    zeroline=False,
                    range=[0, 100]
                ),
                showlegend=False,
                margin=dict(t=60, b=40, l=60, r=20)
            )
            
            viz[f'{category}_concentration'] = fig_concentration
    
    return viz

def create_excel_output(results, stats):
    """Create comprehensive Excel output with multiple sheets"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'bg_color': '#333333',
            'font_color': 'white',
            'border': 1
        })
        
        top_10_format = workbook.add_format({
            'bg_color': '#E8F4F8',
            'border': 1
        })
        
        top_3_formats = [
            workbook.add_format({'bg_color': '#FFD700', 'border': 1}),  # Gold
            workbook.add_format({'bg_color': '#C0C0C0', 'border': 1}),  # Silver
            workbook.add_format({'bg_color': '#CD7F32', 'border': 1})   # Bronze
        ]
        
        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Customers',
                'Total Quantity Delivered',
                'Unique SKUs',
                'Average Order Size',
                'Categories Analyzed',
                'Top Customer',
                'Top Customer Volume',
                'Top 10 Customers Share'
            ],
            'Value': [
                stats.get('total_customers', 0),
                f"{stats.get('total_quantity', 0):,.0f}",
                len(stats.get('unique_skus', [])),
                f"{stats.get('avg_order_size', 0):,.1f}",
                ', '.join([cat.title() for cat in results.keys()]),
                '',  # Will be filled below
                '',  # Will be filled below
                ''   # Will be filled below
            ]
        }
        
        # Get top customer info
        if results:
            for category, df in results.items():
                if df is not None and not df.empty:
                    top_customer = df.iloc[0]
                    customer_col = 'Company' if 'Company' in df.columns else 'Customer'
                    summary_data['Value'][5] = top_customer[customer_col]
                    summary_data['Value'][6] = f"{top_customer['Total Quantity']:,.0f}"
                    
                    top_10_total = df.head(10)['Total Quantity'].sum()
                    total_qty = df['Total Quantity'].sum()
                    summary_data['Value'][7] = f"{(top_10_total/total_qty*100):.1f}%"
                    break
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        worksheet = writer.sheets['Summary']
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:B', 25)
        
        # Add header format
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Category sheets with ALL customers
        for category, df in results.items():
            if df is not None and not df.empty:
                # Create a copy for export
                export_df = df.copy()
                
                # Ensure we have rank column
                if 'Rank' not in export_df.columns:
                    export_df.insert(0, 'Rank', range(1, len(export_df) + 1))
                
                # Add tier classification
                export_df['Tier'] = pd.cut(
                    export_df['Rank'],
                    bins=[0, len(df)*0.2, len(df)*0.5, len(df)],
                    labels=['A', 'B', 'C']
                )
                
                # Add percentage of total
                export_df['% of Total'] = (export_df['Total Quantity'] / export_df['Total Quantity'].sum() * 100).round(2)
                
                # Write to Excel
                sheet_name = category.title()[:31]  # Excel sheet name limit
                export_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                worksheet = writer.sheets[sheet_name]
                
                # Format columns
                customer_col = 'Company' if 'Company' in export_df.columns else 'Customer'
                col_widths = {
                    'Rank': 8,
                    customer_col: 50,
                    'Total Quantity': 15,
                    'SKUs': 60,
                    'SKU Count': 12,
                    'Tier': 8,
                    '% of Total': 12,
                    'Cumulative %': 15
                }
                
                for col_idx, col_name in enumerate(export_df.columns):
                    if col_name in col_widths:
                        worksheet.set_column(col_idx, col_idx, col_widths[col_name])
                
                # Add header format
                for col_num, value in enumerate(export_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Highlight top 10 customers
                for row in range(1, min(11, len(export_df) + 1)):
                    if row <= 3:
                        # Top 3 get special colors
                        row_format = top_3_formats[row - 1]
                    else:
                        # Rest of top 10
                        row_format = top_10_format
                    
                    for col in range(len(export_df.columns)):
                        worksheet.write(row, col, export_df.iloc[row-1, col], row_format)
                
                # Add conditional formatting for tiers
                tier_col = export_df.columns.get_loc('Tier')
                if tier_col >= 0:
                    # Format tier A cells
                    worksheet.conditional_format(11, tier_col, len(export_df), tier_col, {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'A',
                        'format': workbook.add_format({'bg_color': '#90EE90'})
                    })
        
        # Create "All Customers Combined" sheet
        combined_df = pd.DataFrame()
        for category, df in results.items():
            if df is not None and not df.empty:
                df_copy = df.copy()
                df_copy['Category'] = category.title()
                # Standardize column names
                if 'Company' in df_copy.columns:
                    df_copy = df_copy.rename(columns={'Company': 'Customer'})
                combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
        
        if not combined_df.empty:
            # Sort by total quantity
            combined_df = combined_df.sort_values('Total Quantity', ascending=False)
            combined_df['Overall Rank'] = range(1, len(combined_df) + 1)
            
            # Reorder columns
            cols = ['Overall Rank', 'Category', 'Customer', 'Total Quantity']
            if 'SKUs' in combined_df.columns:
                cols.append('SKUs')
            if 'SKU Count' in combined_df.columns:
                cols.append('SKU Count')
            
            combined_df = combined_df[cols]
            combined_df.to_excel(writer, sheet_name='All Customers Combined', index=False)
            
            worksheet = writer.sheets['All Customers Combined']
            worksheet.set_column('A:A', 12)  # Rank
            worksheet.set_column('B:B', 15)  # Category
            worksheet.set_column('C:C', 50)  # Customer
            worksheet.set_column('D:D', 15)  # Quantity
            if 'SKUs' in combined_df.columns:
                worksheet.set_column('E:E', 60)  # SKUs
            
            # Add header format
            for col_num, value in enumerate(combined_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
    
    output.seek(0)
    return output

def display_category_results(category, df):
    """Display results for a single category"""
    st.markdown(f"### {category.title()} Analysis")
    
    # Ensure required columns exist
    if 'Total Quantity' not in df.columns:
        st.error("Missing 'Total Quantity' column in data")
        return
    
    # Key insights cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Customers", 
            f"{len(df):,}",
            help="Total number of unique customers"
        )
    
    with col2:
        st.metric(
            "Total Volume", 
            f"{df['Total Quantity'].sum():,.0f}",
            help="Total quantity delivered"
        )
    
    with col3:
        total_sum = df['Total Quantity'].sum()
        if total_sum > 0:
            top_10_pct = (df.head(10)['Total Quantity'].sum() / total_sum * 100)
        else:
            top_10_pct = 0
        st.metric(
            "Top 10 Concentration", 
            f"{top_10_pct:.1f}%",
            help="Volume share of top 10 customers"
        )
    
    with col4:
        # Find 80% concentration point
        customers_for_80 = 0
        if 'Cumulative %' in df.columns and len(df) > 0:
            customers_for_80 = len(df[df['Cumulative %'] <= 80])
        concentration_pct = (customers_for_80 / len(df) * 100) if len(df) > 0 else 0
        st.metric(
            "80% Volume", 
            f"{customers_for_80} customers",
            f"{concentration_pct:.1f}% of base",
            help="Number of customers contributing 80% of volume"
        )
    
    # Insights section
    st.markdown("#### 💡 Key Insights")
    
    # Calculate tier distribution with safety checks
    total_qty = df['Total Quantity'].sum()
    if total_qty > 0 and len(df) > 0:
        cumulative = 0
        a_customers = b_customers = c_customers = 0
        
        # Ensure df is sorted
        df_sorted = df.sort_values('Total Quantity', ascending=False)
        
        for idx, row in df_sorted.iterrows():
            cumulative += row['Total Quantity']
            if cumulative <= total_qty * 0.2:
                a_customers += 1
            elif cumulative <= total_qty * 0.5:
                b_customers += 1
            else:
                c_customers += 1
    else:
        a_customers = b_customers = c_customers = 0
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="insight-card">
            <strong>Customer Segmentation:</strong><br>
            • Tier A (Strategic): {a_customers} customers<br>
            • Tier B (Important): {b_customers} customers<br>
            • Tier C (Standard): {c_customers} customers
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Top customer info - with safety checks
        customer_col = 'Company' if 'Company' in df.columns else 'Customer'
        if len(df) > 0 and total_qty > 0:
            top_customer = df.iloc[0]
            customer_share = (top_customer['Total Quantity'] / total_qty * 100)
            st.markdown(f"""
            <div class="insight-card">
                <strong>Top Customer:</strong><br>
                {top_customer.get(customer_col, 'N/A')}<br>
                Volume: {top_customer['Total Quantity']:,.0f}<br>
                Share: {customer_share:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-card">
                <strong>Top Customer:</strong><br>
                No customer data available
            </div>
            """, unsafe_allow_html=True)
    
    # Top customers table
    st.markdown("#### 🏆 Top 10 Customers")
    
    top_10_df = df.head(10).copy()
    display_cols = ['Rank']
    display_cols.append(customer_col)
    display_cols.extend(['Total Quantity', 'SKU Count'])
    
    # Select only available columns
    display_cols = [col for col in display_cols if col in top_10_df.columns]
    top_10_display = top_10_df[display_cols].copy()
    
    # Format numbers
    top_10_display['Total Quantity'] = top_10_display['Total Quantity'].apply(lambda x: f'{x:,.0f}')
    
    # Display with custom styling
    st.dataframe(
        top_10_display,
        use_container_width=True,
        height=400,
        hide_index=True
    )
    
    # All customers expandable
    with st.expander("📊 View All Customers", expanded=False):
        # Ensure display_cols are available in df
        available_display_cols = [col for col in display_cols if col in df.columns]
        if available_display_cols:
            all_display = df[available_display_cols].copy()
            if 'Total Quantity' in all_display.columns:
                all_display['Total Quantity'] = all_display['Total Quantity'].apply(lambda x: f'{x:,.0f}')
            
            st.dataframe(
                all_display,
                use_container_width=True,
                height=600,
                hide_index=True
            )
        else:
            st.warning("No data available to display")

def main():
    # Modern header
    st.markdown("""
    <div class="main-header">
        <h1>🎨 Customer Volume Analysis</h1>
        <p style="font-size: 1.125rem; color: #6b7280; margin: 0;">
            Embroidery & Imprinting Customer Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload section with better styling
    uploaded_file = st.file_uploader(
        "Upload Excel Analysis File",
        type=['xlsx', 'xls'],
        help="Upload your hierarchical sales data file with SKUs and customer information"
    )
    
    if uploaded_file:
        # File info display
        col1, col2, col3 = st.columns(3)
        
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        
        with col1:
            st.metric("File Size", f"{file_size:.1f} MB")
        with col2:
            st.metric("File Type", uploaded_file.name.split('.')[-1].upper())
        with col3:
            upload_time = datetime.now().strftime("%H:%M")
            st.metric("Upload Time", upload_time)
        
        try:
            with st.spinner('Processing your file...'):
                # Read file content
                file_content = uploaded_file.read()
                
                # Process the file
                start_time = time.time()
                results, processing_stats = process_excel_file(file_content)
                processing_time = time.time() - start_time
            
            # Display results
            if results:
                # Success message
                st.success(f"✅ Analysis complete in {processing_time:.1f} seconds")
                
                # Display visualizations first
                st.markdown("### 📊 Customer Analytics Dashboard")
                
                viz = create_visualizations(results)
                
                # Display charts for each category
                for category in results.keys():
                    if f'{category}_donut' in viz and f'{category}_concentration' in viz:
                        st.markdown(f"#### {category.title()} Insights")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(viz[f'{category}_donut'], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(viz[f'{category}_concentration'], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed analysis section
                if len(results) > 1:
                    tabs = st.tabs([cat.title() for cat in results.keys()])
                    
                    for idx, (category, df) in enumerate(results.items()):
                        with tabs[idx]:
                            display_category_results(category, df)
                else:
                    # Single category
                    category, df = next(iter(results.items()))
                    display_category_results(category, df)
                
                # Export section with better styling
                st.markdown("---")
                st.markdown("### 💾 Export Your Analysis")
                
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    # Excel download
                    excel_output = create_excel_output(results, processing_stats)
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_output,
                        file_name=f"customer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # CSV download (combined)
                    combined_df = pd.DataFrame()
                    for category, df in results.items():
                        df_copy = df.copy()
                        df_copy['Category'] = category.title()
                        if 'Company' in df_copy.columns:
                            df_copy = df_copy.rename(columns={'Company': 'Customer'})
                        combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
                    
                    if not combined_df.empty:
                        csv = combined_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV Data",
                            data=csv,
                            file_name=f"customer_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            else:
                st.error("❌ No data could be extracted from the file")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Welcome screen with instructions
        st.markdown("""
        <div class="info-card">
            <h4>Welcome to Customer Volume Analysis</h4>
            <p>This tool helps you analyze customer purchase patterns and identify strategic accounts.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4>📤 Upload</h4>
                <p>Upload your hierarchical Excel file with SKU and customer data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4>📊 Analyze</h4>
                <p>Automatically segment customers and identify volume concentration</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
                <h4>💾 Export</h4>
                <p>Download comprehensive Excel report with tier classifications</p>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📋 File Format Requirements", expanded=False):
            st.markdown("""
            **Your Excel file should have:**
            - SKUs in brackets like `[RHB1068GRYIMP] Product Name`
            - Customer names indented under each SKU
            - Total quantities in the rightmost column
            - Sheet names containing 'Imprint' or 'Embroid'
            
            **Example structure:**
            ```
            [SKU123] Product Name                     | Total
            ------------------------------------------|-------
                Customer ABC Company                  | 250
                Customer XYZ Corp                     | 175
            [SKU456] Another Product                  | 
                Customer 123 Inc                      | 300
            ```
            """)

if __name__ == "__main__":
    main()
