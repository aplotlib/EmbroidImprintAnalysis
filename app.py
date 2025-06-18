import pandas as pd
import streamlit as st
import io
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import gc
import numpy as np

# Configure page
st.set_page_config(
    page_title="Embroid/Imprint Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalist design
st.markdown("""
<style>
    /* Minimalist theme */
    .stApp {
        background-color: #fafafa;
    }
    
    /* Clean headers */
    h1, h2, h3 {
        font-weight: 300;
        color: #333;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Clean buttons */
    .stButton > button {
        background-color: #333;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        font-weight: 400;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #555;
        transform: translateY(-1px);
    }
    
    /* Tables */
    .dataframe {
        font-size: 14px;
        border: none !important;
    }
    
    /* Info boxes */
    .info-box {
        background: #f5f5f5;
        border-left: 4px solid #333;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

def detect_file_format(excel_file):
    """
    Detect the format of the Excel file (Sales Analysis or Summary format)
    """
    sheet_names = excel_file.sheet_names
    
    # Check for original sales analysis format
    if any('Sales Analysis' in sheet for sheet in sheet_names):
        return 'sales_analysis'
    
    # Check for summary format - enhanced detection
    elif any('Top Customers' in sheet for sheet in sheet_names):
        return 'summary'
    
    # Check filename patterns if passed
    filename = getattr(excel_file, 'io', None)
    if filename and hasattr(filename, 'name'):
        filename_lower = filename.name.lower()
        if 'top customer' in filename_lower or 'top volume' in filename_lower:
            return 'summary'
    
    # Check for sheets with embroid/imprint in name
    elif any(any(keyword in sheet.lower() for keyword in ['embroid', 'imprint', 'embroider']) 
             for sheet in sheet_names):
        return 'summary'  # Treat as summary format
    
    # Check for generic sheets that might contain customer data
    elif any(sheet.lower() in ['imprinting', 'embroidery', 'customers', 'data', 'sheet1'] 
             for sheet in sheet_names):
        return 'generic'
    
    return 'unknown'

def process_summary_format(excel_file, file_content):
    """
    Process files that are already in summary format
    """
    results = {}
    processing_stats = {
        'total_customers': 0,
        'total_skus': set(),
        'total_quantity': 0,
        'avg_order_size': 0,
        'sheets_processed': 0,
        'empty_sheets': []
    }
    
    # Check if we need to assign categories based on filename
    filename = getattr(excel_file, 'filename', '')
    has_both_categories = False
    if filename:
        filename_lower = filename.lower()
        has_both_categories = 'embroid' in filename_lower and 'imprint' in filename_lower
    
    # Process each sheet
    for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
        try:
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name, engine='openpyxl')
            
            # Skip empty sheets
            if df.empty or len(df) == 0:
                processing_stats['empty_sheets'].append(sheet_name)
                continue
            
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty or len(df) == 0:
                processing_stats['empty_sheets'].append(sheet_name)
                continue
            
            processing_stats['sheets_processed'] += 1
            
            # Detect column names (case-insensitive)
            col_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower().strip()
                # Enhanced customer column detection
                if 'customer' in col_lower or 'company' in col_lower or 'client' in col_lower or 'account' in col_lower:
                    col_mapping['customer'] = col
                # Enhanced quantity column detection
                elif any(keyword in col_lower for keyword in ['qty', 'quantity', 'amount', 'volume', 'total', 'units']):
                    # Make sure it's not SKU count
                    if 'sku' not in col_lower:
                        col_mapping['qty'] = col
                # SKU list/ordered column
                elif 'sku' in col_lower and any(keyword in col_lower for keyword in ['ordered', 'list', 'items', 'products']):
                    col_mapping['skus'] = col
                elif 'sku' in col_lower and 'count' in col_lower:
                    col_mapping['sku_count'] = col
                elif 'sku' in col_lower and col_mapping.get('skus') is None:
                    col_mapping['skus'] = col
                # Additional column patterns
                elif 'product' in col_lower and 'ordered' in col_lower:
                    col_mapping['skus'] = col
            
            # If no customer column found, check first column
            if 'customer' not in col_mapping and len(df.columns) > 0:
                first_col = df.columns[0]
                # Check if first column contains text that looks like customer names
                if df[first_col].dtype == 'object':
                    non_null_values = df[first_col].dropna()
                    if len(non_null_values) > 0:
                        col_mapping['customer'] = first_col
            
            # If no qty column found, look for numeric columns
            if 'qty' not in col_mapping:
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        # Check if this column has positive values
                        if (df[col] > 0).any():
                            col_mapping['qty'] = col
                            break
            
            # Ensure we have minimum required columns
            if 'customer' in col_mapping and 'qty' in col_mapping:
                # Standardize column names
                df_standard = pd.DataFrame()
                df_standard['Customer'] = df[col_mapping['customer']].astype(str)
                df_standard['Qty Delivered'] = pd.to_numeric(df[col_mapping['qty']], errors='coerce').fillna(0).astype(int)
                
                # Add SKU information if available
                if 'skus' in col_mapping:
                    df_standard['SKUs Ordered'] = df[col_mapping['skus']].fillna('').astype(str)
                    # Extract SKU count
                    df_standard['SKU Count'] = df_standard['SKUs Ordered'].apply(
                        lambda x: len([s.strip() for s in str(x).split(',') if s.strip()]) if x and x != 'nan' else 0
                    )
                elif 'sku_count' in col_mapping:
                    df_standard['SKU Count'] = pd.to_numeric(df[col_mapping['sku_count']], errors='coerce').fillna(0).astype(int)
                    df_standard['SKUs Ordered'] = ''
                else:
                    df_standard['SKU Count'] = 1  # Assume at least 1 SKU per customer
                    df_standard['SKUs Ordered'] = ''
                
                # Calculate average order size
                df_standard['Avg Order Size'] = (df_standard['Qty Delivered'] / df_standard['SKU Count'].clip(lower=1)).round(1)
                
                # Remove any rows with invalid customer names or zero quantity
                df_standard = df_standard[df_standard['Customer'].notna()]
                df_standard = df_standard[df_standard['Customer'].str.strip() != '']
                df_standard = df_standard[df_standard['Customer'] != 'nan']
                df_standard = df_standard[df_standard['Qty Delivered'] > 0]
                
                if len(df_standard) == 0:
                    continue
                
                # Sort by quantity
                df_standard = df_standard.sort_values('Qty Delivered', ascending=False).reset_index(drop=True)
                
                # Calculate Pareto analysis
                if len(df_standard) > 0 and df_standard['Qty Delivered'].sum() > 0:
                    df_standard['Cumulative %'] = (df_standard['Qty Delivered'].cumsum() / df_standard['Qty Delivered'].sum() * 100).round(1)
                    df_standard['Revenue Tier'] = pd.cut(
                        df_standard['Cumulative %'], 
                        bins=[0, 80, 95, 100], 
                        labels=['A (Top 80%)', 'B (Next 15%)', 'C (Bottom 5%)']
                    )
                
                # Update stats
                processing_stats['total_quantity'] += int(df_standard['Qty Delivered'].sum())
                processing_stats['total_customers'] += len(df_standard)
                
                # Extract unique SKUs
                for skus in df_standard['SKUs Ordered']:
                    if skus and skus != 'nan' and skus.strip():
                        for sku in str(skus).split(','):
                            sku = sku.strip()
                            if sku:
                                processing_stats['total_skus'].add(sku)
                
                # Determine category based on sheet name or assign generically
                sheet_lower = sheet_name.lower()
                # Enhanced category detection
                if any(keyword in sheet_lower for keyword in ['imprint', 'imprnt', 'imp ']):
                    results['imprinting'] = df_standard
                elif any(keyword in sheet_lower for keyword in ['embroid', 'emb ', 'embro']):
                    df_standard = df_standard.rename(columns={'Customer': 'Company'})
                    results['embroidery'] = df_standard
                elif has_both_categories:
                    # File contains both categories, assign based on sheet order or what's empty
                    if sheet_idx == 0 or 'imprinting' not in results:
                        results['imprinting'] = df_standard
                    else:
                        df_standard = df_standard.rename(columns={'Customer': 'Company'})
                        results['embroidery'] = df_standard
                else:
                    # For generic sheet names (Sheet1, Sheet2, etc.)
                    # Try to determine from any clues in the data
                    embroid_found = False
                    imprint_found = False
                    
                    # Check SKU patterns
                    if 'SKUs Ordered' in df_standard.columns:
                        all_skus = ' '.join(df_standard['SKUs Ordered'].astype(str))
                        if any(keyword in all_skus.lower() for keyword in ['emb', 'embroid']):
                            embroid_found = True
                        if any(keyword in all_skus.lower() for keyword in ['imp', 'imprint']):
                            imprint_found = True
                    
                    # Assign based on what was found
                    if imprint_found and not embroid_found:
                        results['imprinting'] = df_standard
                    elif embroid_found and not imprint_found:
                        df_standard = df_standard.rename(columns={'Customer': 'Company'})
                        results['embroidery'] = df_standard
                    else:
                        # Default assignment - alternate between categories
                        if 'imprinting' not in results:
                            results['imprinting'] = df_standard
                        else:
                            df_standard = df_standard.rename(columns={'Customer': 'Company'})
                            results['embroidery'] = df_standard
        
        except Exception as e:
            st.warning(f"Could not process sheet '{sheet_name}': {str(e)}")
            continue
    
    # Calculate average order size
    if processing_stats['total_customers'] > 0:
        processing_stats['avg_order_size'] = round(
            processing_stats['total_quantity'] / processing_stats['total_customers'], 1
        )
    
    return results, processing_stats

@st.cache_data
def process_customer_sku_data(file_content, filename):
    """
    Process Excel file with auto-format detection.
    Enhanced to handle various file naming conventions.
    """
    # Read Excel file from bytes
    excel_file = pd.ExcelFile(io.BytesIO(file_content))
    
    # Store filename for later use in format detection
    excel_file.filename = filename
    
    # Detect file format
    file_format = detect_file_format(excel_file)
    
    # If format is unknown, check filename patterns
    if file_format == 'unknown' and filename:
        filename_lower = filename.lower()
        if any(keyword in filename_lower for keyword in ['top customer', 'top volume', 'customer list']):
            file_format = 'summary'
    
    if file_format == 'summary' or file_format == 'generic':
        return process_summary_format(excel_file, file_content)
    
    # Original processing for sales analysis format
    results = {}
    processing_stats = {
        'total_customers': 0,
        'total_skus': set(),
        'total_quantity': 0,
        'avg_order_size': 0
    }
    
    # Try different sheet name patterns
    sheet_patterns = [
        ['Imprinting - Sales Analysis', 'Embroidery - Sales Analysis'],  # Original format
        ['Imprinting', 'Embroidery'],  # Simplified names
        excel_file.sheet_names  # All sheets if patterns don't match
    ]
    
    sheets_processed = False
    
    for patterns in sheet_patterns:
        for sheet_name in patterns:
            if sheet_name in excel_file.sheet_names:
                try:
                    # Read with optimization for large files
                    df = pd.read_excel(
                        io.BytesIO(file_content), 
                        sheet_name=sheet_name,
                        engine='openpyxl'
                    )
                    
                    # Skip empty sheets
                    if df.empty or len(df) < 5:
                        continue
                    
                    # Process data
                    customer_data = defaultdict(lambda: {
                        'total_qty': 0, 
                        'skus': set(),
                        'order_count': 0,
                        'avg_order_size': 0
                    })
                    current_sku = None
                    
                    # Skip header rows more efficiently
                    data_start_idx = 4
                    
                    for idx in range(data_start_idx, len(df)):
                        row = df.iloc[idx]
                        col_a_value = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
                        
                        # SKU detection
                        if '[' in col_a_value and ']' in col_a_value:
                            sku_start = col_a_value.find('[')
                            sku_end = col_a_value.find(']')
                            current_sku = col_a_value[sku_start:sku_end+1] + col_a_value[sku_end+1:].strip()
                            processing_stats['total_skus'].add(current_sku)
                        else:
                            customer_name = col_a_value.strip()
                            
                            if customer_name and customer_name != 'Total' and current_sku:
                                try:
                                    # Try column H (index 7) first, then look for numeric columns
                                    if len(row) > 7:
                                        qty = float(row.iloc[7]) if pd.notna(row.iloc[7]) else 0
                                    else:
                                        # Find first numeric column
                                        qty = 0
                                        for i in range(1, len(row)):
                                            try:
                                                val = float(row.iloc[i])
                                                if val > 0:
                                                    qty = val
                                                    break
                                            except:
                                                continue
                                except:
                                    qty = 0
                                
                                if qty > 0:
                                    customer_data[customer_name]['total_qty'] += qty
                                    customer_data[customer_name]['skus'].add(current_sku)
                                    customer_data[customer_name]['order_count'] += 1
                                    processing_stats['total_quantity'] += qty
                    
                    # Convert to DataFrame with additional analytics
                    if customer_data:
                        data_list = []
                        for customer, data in customer_data.items():
                            avg_order = data['total_qty'] / data['order_count'] if data['order_count'] > 0 else 0
                            data_list.append({
                                'Customer': customer,
                                'Qty Delivered': int(data['total_qty']),
                                'SKU Count': len(data['skus']),
                                'Avg Order Size': round(avg_order, 1),
                                'SKUs Ordered': ', '.join(sorted(data['skus']))
                            })
                        
                        # Create sorted DataFrame
                        result_df = pd.DataFrame(data_list)
                        result_df = result_df.sort_values('Qty Delivered', ascending=False)
                        result_df = result_df.reset_index(drop=True)
                        
                        # Calculate Pareto (80/20 rule)
                        result_df['Cumulative %'] = (result_df['Qty Delivered'].cumsum() / result_df['Qty Delivered'].sum() * 100).round(1)
                        result_df['Revenue Tier'] = pd.cut(
                            result_df['Cumulative %'], 
                            bins=[0, 80, 95, 100], 
                            labels=['A (Top 80%)', 'B (Next 15%)', 'C (Bottom 5%)']
                        )
                        
                        # Store results
                        processing_stats['total_customers'] += len(result_df)
                        
                        # Determine category
                        sheet_lower = sheet_name.lower()
                        if 'imprint' in sheet_lower:
                            results['imprinting'] = result_df
                        elif 'embroid' in sheet_lower:
                            result_df = result_df.rename(columns={'Customer': 'Company'})
                            results['embroidery'] = result_df
                        else:
                            # Generic assignment
                            if 'imprinting' not in results:
                                results['imprinting'] = result_df
                            else:
                                result_df = result_df.rename(columns={'Customer': 'Company'})
                                results['embroidery'] = result_df
                        
                        sheets_processed = True
                
                except Exception as e:
                    st.warning(f"Could not process sheet '{sheet_name}': {str(e)}")
                    continue
        
        if sheets_processed:
            break
    
    # If no valid data found, try generic processing
    if not results and excel_file.sheet_names:
        # Process first sheet as generic customer data
        df = pd.read_excel(io.BytesIO(file_content), sheet_name=0, engine='openpyxl')
        if not df.empty:
            # Try to identify customer and quantity columns
            results, processing_stats = process_summary_format(excel_file, file_content)
    
    # Calculate average order size
    if processing_stats['total_customers'] > 0:
        processing_stats['avg_order_size'] = round(
            processing_stats['total_quantity'] / processing_stats['total_customers'], 1
        )
    
    # Clean up memory for large files
    gc.collect()
    
    return results, processing_stats

def validate_and_diagnose_file(file_content):
    """
    Validate file and provide diagnostic information
    """
    diagnostics = {
        'file_readable': False,
        'sheets': [],
        'issues': [],
        'recommendations': []
    }
    
    try:
        excel_file = pd.ExcelFile(io.BytesIO(file_content))
        diagnostics['file_readable'] = True
        
        for sheet_name in excel_file.sheet_names:
            sheet_info = {
                'name': sheet_name,
                'rows': 0,
                'cols': 0,
                'has_data': False,
                'potential_customer_col': None,
                'potential_qty_col': None,
                'detected_columns': []
            }
            
            try:
                df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
                sheet_info['rows'] = len(df)
                sheet_info['cols'] = len(df.columns)
                
                if not df.empty:
                    # Remove completely empty rows/cols
                    df_clean = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if not df_clean.empty:
                        sheet_info['has_data'] = True
                        
                        # Enhanced column detection
                        for col in df_clean.columns:
                            col_str = str(col).strip()
                            col_lower = col_str.lower()
                            
                            # Customer column detection
                            if any(keyword in col_lower for keyword in ['customer', 'company', 'client', 'account', 'name']):
                                sheet_info['potential_customer_col'] = col_str
                                sheet_info['detected_columns'].append(f"Customer: {col_str}")
                            
                            # Quantity column detection
                            elif any(keyword in col_lower for keyword in ['qty', 'quantity', 'volume', 'units', 'total']):
                                if 'sku' not in col_lower:  # Avoid SKU count columns
                                    sheet_info['potential_qty_col'] = col_str
                                    sheet_info['detected_columns'].append(f"Quantity: {col_str}")
                            
                            # SKU column detection
                            elif 'sku' in col_lower:
                                sheet_info['detected_columns'].append(f"SKU: {col_str}")
                            
                            # Additional pattern: first text column might be customer
                            elif not sheet_info['potential_customer_col'] and df_clean[col].dtype == 'object':
                                non_null = df_clean[col].dropna()
                                if len(non_null) > 3:  # Has enough data
                                    # Check if values look like company names
                                    sample_values = non_null.head(5).astype(str)
                                    if all(len(val) > 3 and not val.replace('.','').replace(',','').isdigit() 
                                          for val in sample_values):
                                        sheet_info['potential_customer_col'] = col_str
                                        sheet_info['detected_columns'].append(f"Likely Customer: {col_str}")
                
            except Exception as e:
                diagnostics['issues'].append(f"Could not read sheet '{sheet_name}': {str(e)}")
            
            diagnostics['sheets'].append(sheet_info)
        
        # Generate recommendations
        total_sheets = len(diagnostics['sheets'])
        sheets_with_data = sum(1 for s in diagnostics['sheets'] if s['has_data'])
        
        if sheets_with_data == 0:
            diagnostics['recommendations'].append("All sheets appear to be empty. Please check if the file was exported correctly.")
            diagnostics['recommendations'].append("Try opening the file in Excel to verify it contains data.")
        elif sheets_with_data < total_sheets:
            empty_sheets = [s['name'] for s in diagnostics['sheets'] if not s['has_data']]
            diagnostics['recommendations'].append(f"Some sheets are empty: {', '.join(empty_sheets)}")
        
        # Check for customer/qty columns
        sheets_missing_customer = [s['name'] for s in diagnostics['sheets'] 
                                 if s['has_data'] and not s['potential_customer_col']]
        sheets_missing_qty = [s['name'] for s in diagnostics['sheets'] 
                            if s['has_data'] and not s['potential_qty_col']]
        
        if sheets_missing_customer:
            diagnostics['recommendations'].append(f"Could not identify customer column in: {', '.join(sheets_missing_customer)}")
            diagnostics['recommendations'].append("Ensure customer names are in a column labeled 'Customer', 'Company', or similar")
        if sheets_missing_qty:
            diagnostics['recommendations'].append(f"Could not identify quantity column in: {', '.join(sheets_missing_qty)}")
            diagnostics['recommendations'].append("Ensure quantities are in a column labeled 'Qty', 'Quantity', 'Volume', or similar")
            
    except Exception as e:
        diagnostics['issues'].append(f"Could not read file: {str(e)}")
        diagnostics['recommendations'].append("File may be corrupted or in an unsupported format.")
        diagnostics['recommendations'].append("Try saving the file as .xlsx format in Excel.")
    
    return diagnostics

def create_visualizations(results):
    """Create clean, minimalist visualizations"""
    viz = {}
    
    # Color palette
    colors = ['#333333', '#666666', '#999999', '#CCCCCC', '#E5E5E5']
    
    for category, df in results.items():
        if df is not None and len(df) > 0:
            # Top 10 customers bar chart
            top_10 = df.head(10).copy()
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=top_10['Qty Delivered'],
                y=top_10['Customer' if category == 'imprinting' else 'Company'],
                orientation='h',
                marker_color='#333333',
                text=top_10['Qty Delivered'],
                textposition='auto',
            ))
            
            fig_bar.update_layout(
                title=f"Top 10 {category.title()} Customers",
                xaxis_title="Quantity Delivered",
                yaxis_title="",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12, color="#333"),
                height=400,
                margin=dict(l=200, r=20, t=40, b=40),
                showlegend=False
            )
            
            fig_bar.update_xaxis(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig_bar.update_yaxis(showgrid=False)
            
            # Pareto chart
            fig_pareto = go.Figure()
            
            # Bar trace
            fig_pareto.add_trace(go.Bar(
                x=df.index[:20],
                y=df['Qty Delivered'][:20],
                name='Quantity',
                marker_color='#e0e0e0',
                yaxis='y'
            ))
            
            # Line trace
            fig_pareto.add_trace(go.Scatter(
                x=df.index[:20],
                y=df['Cumulative %'][:20],
                name='Cumulative %',
                mode='lines+markers',
                line=dict(color='#333', width=2),
                marker=dict(size=6),
                yaxis='y2'
            ))
            
            fig_pareto.update_layout(
                title=f"{category.title()} Pareto Analysis (80/20 Rule)",
                xaxis_title="Customer Rank",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12, color="#333"),
                height=400,
                yaxis=dict(title='Quantity', showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
                yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
                showlegend=True,
                legend=dict(x=0.7, y=0.95)
            )
            
            viz[f'{category}_bar'] = fig_bar
            viz[f'{category}_pareto'] = fig_pareto
    
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
        
        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Customers',
                'Total Quantity Delivered',
                'Unique SKUs',
                'Average Order Size',
                'Processing Date'
            ],
            'Value': [
                stats['total_customers'],
                f"{stats['total_quantity']:,}",
                len(stats['total_skus']),
                stats['avg_order_size'],
                datetime.now().strftime('%Y-%m-%d %H:%M')
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format summary sheet
        worksheet = writer.sheets['Summary']
        for i, col in enumerate(summary_df.columns):
            worksheet.write(0, i, col, header_format)
            worksheet.set_column(i, i, 30)
        
        # Write detailed data sheets
        for category, df in results.items():
            if df is not None:
                sheet_name = f'{category.title()} Analysis'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                worksheet = writer.sheets[sheet_name]
                
                # Format headers
                for i, col in enumerate(df.columns):
                    worksheet.write(0, i, col, header_format)
                    
                    # Set column widths
                    if col == 'SKUs Ordered':
                        worksheet.set_column(i, i, 80)
                    elif col in ['Customer', 'Company']:
                        worksheet.set_column(i, i, 40)
                    else:
                        worksheet.set_column(i, i, 15)
                
                # Add conditional formatting for Revenue Tier
                if 'Revenue Tier' in df.columns:
                    tier_col = df.columns.get_loc('Revenue Tier')
                    worksheet.conditional_format(1, tier_col, len(df), tier_col, {
                        'type': 'text',
                        'criteria': 'containing',
                        'value': 'A',
                        'format': workbook.add_format({'bg_color': '#90EE90'})
                    })
    
    output.seek(0)
    return output

def main():
    # Header
    st.markdown("# 🎨 Embroid/Imprint Analysis")
    st.markdown("Upload sales analysis files to identify top customers and SKU patterns")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Select Excel file",
        type=['xlsx', 'xls'],
        help="Supports: Sales Analysis format, Customer Summary format, and files up to 200MB"
    )
    
    if uploaded_file:
        # Check file size
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        
        if file_size > 200:
            st.error("File size exceeds 200MB limit. Please split the file or contact support.")
            return
        
        try:
            # Show processing status
            with st.spinner(f'Processing {uploaded_file.name} ({file_size:.1f} MB)...'):
                # Read file content once
                file_content = uploaded_file.read()
                
                # Process with caching
                results, stats = process_customer_sku_data(file_content, uploaded_file.name)
            
            if not results:
                # Run diagnostics on the file
                diagnostics = validate_and_diagnose_file(file_content)
                
                st.error("No valid data found in the file.")
                
                # Show diagnostic information
                with st.expander("🔍 File Diagnostics", expanded=True):
                    if diagnostics['file_readable']:
                        st.markdown("✅ **File is readable**")
                        
                        # Sheet information
                        st.markdown("\n**Sheet Analysis:**")
                        for sheet in diagnostics['sheets']:
                            status = "✅" if sheet['has_data'] else "❌"
                            st.markdown(f"{status} **{sheet['name']}**: {sheet['rows']} rows, {sheet['cols']} columns")
                            if sheet['has_data']:
                                if sheet.get('detected_columns'):
                                    st.markdown(f"  - Detected: {', '.join(sheet['detected_columns'])}")
                                else:
                                    if sheet['potential_customer_col']:
                                        st.markdown(f"  - Potential customer column: `{sheet['potential_customer_col']}`")
                                    if sheet['potential_qty_col']:
                                        st.markdown(f"  - Potential quantity column: `{sheet['potential_qty_col']}`")
                    else:
                        st.markdown("❌ **File could not be read**")
                    
                    # Issues
                    if diagnostics['issues']:
                        st.markdown("\n**Issues Found:**")
                        for issue in diagnostics['issues']:
                            st.markdown(f"- {issue}")
                    
                    # Recommendations
                    if diagnostics['recommendations']:
                        st.markdown("\n**Recommendations:**")
                        for rec in diagnostics['recommendations']:
                            st.markdown(f"- {rec}")
                
                st.markdown("""
                <div class="info-box">
                <strong>Next Steps:</strong><br>
                1. Verify the file contains data by opening it in Excel<br>
                2. Ensure data is in a standard table format (not pivot tables)<br>
                3. Check that customer names and quantities are in separate columns<br>
                4. Try using the sample file generator below as a template<br>
                5. If the issue persists, try re-exporting from your source system
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Show file info for transparency
            with st.expander("📄 File Detection Info", expanded=False):
                excel_file = pd.ExcelFile(io.BytesIO(file_content))
                excel_file.filename = uploaded_file.name  # Store filename
                format_detected = detect_file_format(excel_file)
                
                st.markdown(f"**File Name:** {uploaded_file.name}")
                st.markdown(f"**File Format Detected:** {format_detected.replace('_', ' ').title()}")
                st.markdown(f"**Sheets Found:** {', '.join(excel_file.sheet_names)}")
                
                # Show column detection info for first sheet with data
                for sheet_name in excel_file.sheet_names[:1]:  # Just first sheet
                    try:
                        df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name, nrows=5)
                        if not df.empty:
                            st.markdown(f"\n**Column Detection for '{sheet_name}':**")
                            
                            # Show detected columns
                            col_info = []
                            for col in df.columns:
                                col_lower = str(col).lower()
                                if any(k in col_lower for k in ['customer', 'company', 'client']):
                                    col_info.append(f"- `{col}` → Customer column")
                                elif any(k in col_lower for k in ['qty', 'quantity', 'volume']):
                                    col_info.append(f"- `{col}` → Quantity column")
                                elif 'sku' in col_lower:
                                    col_info.append(f"- `{col}` → SKU column")
                            
                            if col_info:
                                st.markdown('\n'.join(col_info))
                            
                            st.markdown("\n**Data Preview (first 5 rows):**")
                            st.dataframe(df, use_container_width=True)
                    except:
                        pass
            
            # Success message
            st.success(f"✓ Processed {stats['total_customers']} customers across {len(stats['total_skus'])} unique SKUs")
            
            # Create three columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Quantity", 
                    f"{stats['total_quantity']:,}",
                    help="Total items delivered across all customers"
                )
            
            with col2:
                st.metric(
                    "Unique SKUs", 
                    len(stats['total_skus']),
                    help="Number of different products ordered"
                )
            
            with col3:
                st.metric(
                    "Avg Order Size", 
                    f"{stats['avg_order_size']:.1f}",
                    help="Average quantity per customer"
                )
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "📈 Top Customers", "📋 Full Data", "💾 Export"])
            
            with tab1:
                # Create visualizations
                viz = create_visualizations(results)
                
                # Display charts
                for key in ['imprinting', 'embroidery']:
                    if f'{key}_bar' in viz:
                        st.subheader(f"{key.title()} Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(viz[f'{key}_bar'], use_container_width=True)
                        
                        with col2:
                            st.plotly_chart(viz[f'{key}_pareto'], use_container_width=True)
                        
                        # Key insights
                        df = results[key]
                        top_80_count = len(df[df['Cumulative %'] <= 80])
                        top_80_pct = (top_80_count / len(df) * 100)
                        
                        st.markdown(f"""
                        <div class="info-box">
                        <strong>Key Insights:</strong><br>
                        • Top {top_80_count} customers ({top_80_pct:.1f}%) drive 80% of volume<br>
                        • Most popular SKU: {df.iloc[0]['SKUs Ordered'].split(',')[0] if ',' in df.iloc[0]['SKUs Ordered'] else df.iloc[0]['SKUs Ordered']}<br>
                        • Largest order: {df.iloc[0]['Qty Delivered']:,} units
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                # Top customers comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'imprinting' in results:
                        st.subheader("🖨️ Top Imprinting Customers")
                        display_df = results['imprinting'][['Customer', 'Qty Delivered', 'SKU Count', 'Revenue Tier']].head(20)
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Qty Delivered": st.column_config.NumberColumn(format="%d"),
                                "SKU Count": st.column_config.NumberColumn(format="%d")
                            }
                        )
                
                with col2:
                    if 'embroidery' in results:
                        st.subheader("🧵 Top Embroidery Customers")
                        display_df = results['embroidery'][['Company', 'Qty Delivered', 'SKU Count', 'Revenue Tier']].head(20)
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Qty Delivered": st.column_config.NumberColumn(format="%d"),
                                "SKU Count": st.column_config.NumberColumn(format="%d")
                            }
                        )
            
            with tab3:
                # Full data view with search
                view_option = st.selectbox("Select data to view:", ["Imprinting", "Embroidery"])
                
                if view_option.lower() in results:
                    df = results[view_option.lower()]
                    
                    # Search functionality
                    search = st.text_input("🔍 Search customers or SKUs:", "")
                    
                    if search:
                        mask = (
                            df[df.columns[0]].str.contains(search, case=False, na=False) |
                            df['SKUs Ordered'].str.contains(search, case=False, na=False)
                        )
                        filtered_df = df[mask]
                        st.info(f"Found {len(filtered_df)} matches")
                    else:
                        filtered_df = df
                    
                    # Display with pagination
                    st.dataframe(
                        filtered_df,
                        use_container_width=True,
                        hide_index=True,
                        height=600
                    )
            
            with tab4:
                # Export options
                st.markdown("### 📥 Download Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    excel_output = create_excel_output(results, stats)
                    st.download_button(
                        label="📑 Download Full Analysis (Excel)",
                        data=excel_output,
                        file_name=f"Customer_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col2:
                    # Quick CSV export of top customers
                    if st.button("📊 Generate Top 20 Report", use_container_width=True):
                        csv_data = []
                        for category, df in results.items():
                            if df is not None:
                                top_20 = df.head(20).copy()
                                top_20['Category'] = category.title()
                                csv_data.append(top_20)
                        
                        if csv_data:
                            combined_csv = pd.concat(csv_data, ignore_index=True)
                            csv_str = combined_csv.to_csv(index=False)
                            
                            st.download_button(
                                label="Download Top 20 CSV",
                                data=csv_str,
                                file_name=f"Top_20_Customers_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
            # Provide specific troubleshooting based on error
            if "list index out of range" in str(e):
                st.markdown("""
                <div class="info-box">
                <strong>File structure issue detected. Please check:</strong><br>
                • Column A should contain customer names or SKU information<br>
                • Quantity data should be in column H (for Sales Analysis format)<br>
                • For Summary format: ensure columns are named properly<br>
                • Data should start from row 5 (Sales Analysis) or row 1 (Summary)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                <strong>Troubleshooting:</strong><br>
                • <b>Top Customer files</b>: Files named like "Imp_Embroid Top Customers All Time.xlsx"<br>
                • <b>Volume Customer files</b>: Files named like "Imprint_Embroid top volume customers SKU.xlsx"<br>
                • <b>Summary format</b>: Columns for Customer/Company, Qty Delivered, SKUs Ordered<br>
                • <b>Sales Analysis format</b>: Sheets named 'Imprinting/Embroidery - Sales Analysis'<br>
                • <b>Generic format</b>: Any sheet with customer and quantity columns<br>
                • Ensure the file is not password protected<br>
                • Column names are flexible: Customer/Company/Client, Qty/Quantity/Volume, etc.<br>
                • Try saving in .xlsx format if using .xls
                </div>
                """, unsafe_allow_html=True)
            
            # Show detected sheets for debugging
            try:
                excel_file = pd.ExcelFile(io.BytesIO(file_content))
                st.info(f"Detected sheets: {', '.join(excel_file.sheet_names)}")
            except:
                pass
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        ### 📁 How to use:
        
        1. **Upload** your Excel file containing sales or customer data
        2. **View** automated analysis with customer rankings and SKU patterns  
        3. **Export** comprehensive reports for further analysis
        
        ### 📊 What you'll get:
        
        - **Customer Rankings** - Identify your top volume customers
        - **Pareto Analysis** - See which customers drive 80% of your business
        - **SKU Insights** - Understand product ordering patterns
        - **Revenue Tiers** - Automatic A/B/C customer classification
        
        ### ⚡ Features:
        
        - Handles files up to 200MB
        - Auto-detects multiple file formats
        - Real-time search and filtering
        - Export to Excel with formatting
        """)
        
        # Sample data structure
        with st.expander("📋 Supported File Formats"):
            st.markdown("""
            **Format 1: Sales Analysis**
            - Sheet names: `Imprinting - Sales Analysis` or `Embroidery - Sales Analysis`
            - Customer names in column A
            - SKU information in brackets [SKU123] 
            - Quantities in column H
            - Data starting from row 5
            
            **Format 2: Customer Summary / Top Customers**
            - Files like: "Imp_Embroid Top Customers All Time.xlsx"
            - Files like: "Imprint_Embroid top volume customers SKU.xlsx"
            - Pre-processed customer data
            - Columns: Customer/Company, Qty Delivered, SKUs Ordered
            - Any sheet name containing 'Top Customers', 'Embroid', or 'Imprint'
            
            **Format 3: Generic Format**
            - Any Excel sheet with:
                - Customer, Company, Client, or Account column
                - Quantity, Qty, Volume, Units, or Total column
                - Optional: SKU information
            
            The tool will automatically detect your file format and extract:
            - Customer rankings by volume
            - SKU patterns and popular products
            - Pareto analysis (80/20 rule)
            - Revenue tier classification (A/B/C)
            """)
        
        # Generate sample file
        with st.expander("💾 Generate Sample File"):
            if st.button("Download Sample Excel File"):
                # Create two sample sheets matching user's format
                sample_data_imp = {
                    'Customer': ['ABC Company', 'XYZ Corp', 'Test Inc', 'Demo LLC', 'Sample Co'],
                    'Qty Delivered': [500, 350, 200, 150, 100],
                    'SKUs Ordered': ['SKU001, SKU002', 'SKU001', 'SKU003, SKU004, SKU005', 'SKU002', 'SKU001, SKU003'],
                    'SKU Count': [2, 1, 3, 1, 2]
                }
                sample_data_emb = {
                    'Company': ['Fashion Brand A', 'Retail Chain B', 'Sports Team C', 'Corporate D', 'School E'],
                    'Qty Delivered': [800, 600, 400, 300, 200],
                    'SKUs Ordered': ['EMB-101, EMB-102', 'EMB-101', 'EMB-201, EMB-202', 'EMB-301', 'EMB-101, EMB-201'],
                    'SKU Count': [2, 1, 2, 1, 2]
                }
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Create sheets matching user's naming patterns
                    pd.DataFrame(sample_data_imp).to_excel(writer, sheet_name='Imp Top Customers', index=False)
                    pd.DataFrame(sample_data_emb).to_excel(writer, sheet_name='Embroid Top Customers', index=False)
                
                output.seek(0)
                st.download_button(
                    label="📥 Download Sample File",
                    data=output,
                    file_name="Imp_Embroid_Top_Customers_Sample.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
