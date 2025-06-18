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

# Configure page
st.set_page_config(
    page_title="Embroid/Imprint Analysis",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Early function definitions for safety
def fix_column_types(df):
    """Fix column types to prevent Arrow serialization errors"""
    df = df.copy()  # Work on a copy to avoid warnings
    
    # Special handling for columns literally named 'Total'
    if 'Total' in df.columns:
        # Always convert 'Total' column to numeric or remove it
        try:
            df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
        except:
            # If conversion fails, drop the column
            df = df.drop(columns=['Total'])
            
    for col in df.columns:
        # Skip if we already handled Total
        if col == 'Total' and col not in df.columns:
            continue
            
        # Special handling for columns that might contain mixed types
        if df[col].dtype == 'object' or df[col].dtype.name == 'object':
            # Get non-null values to analyze
            non_null = df[col].dropna()
            
            if len(non_null) == 0:
                # All values are null, convert to string
                df[col] = df[col].fillna('').astype(str)
                continue
            
            # Check what types we have
            numeric_count = 0
            string_count = 0
            
            for val in non_null.head(100):  # Sample first 100 non-null values
                try:
                    if isinstance(val, (int, float)):
                        numeric_count += 1
                    elif isinstance(val, str):
                        # Try to convert string to number
                        try:
                            # Clean common number formats - FIXED THE UNTERMINATED STRING HERE
                            cleaned = str(val).replace(',', '').replace('%', '').strip()
                            if cleaned and cleaned != 'nan':
                                float(cleaned)
                                numeric_count += 1
                            else:
                                string_count += 1
                        except:
                            string_count += 1
                    else:
                        string_count += 1
                except:
                    string_count += 1
            
            # Decide on type based on majority
            if numeric_count > string_count:
                # Convert to numeric
                def safe_numeric_convert(x):
                    if pd.isna(x):
                        return 0
                    try:
                        cleaned = str(x).replace(',', '').replace('%', '').strip()
                        return pd.to_numeric(cleaned, errors='coerce')
                    except:
                        return 0
                
                df[col] = df[col].apply(safe_numeric_convert)
                df[col] = df[col].fillna(0).astype(float)
            else:
                # Convert to string
                df[col] = df[col].fillna('').astype(str)
        
        # Also check for any remaining object dtypes in numeric columns
        elif df[col].dtype in ['int64', 'float64']:
            # Ensure no NaN values that might cause issues
            df[col] = df[col].fillna(0)
    
    return df

def safe_dataframe_display(df, **kwargs):
    """Safely display a dataframe with fallback for Arrow errors"""
    try:
        # First attempt with type fixing
        df_fixed = fix_column_types(df.copy())
        st.dataframe(df_fixed, **kwargs)
    except Exception as e:
        error_msg = str(e)
        if "Arrow" in error_msg or "Serialization" in error_msg or "Expected bytes" in error_msg:
            # More aggressive fix - convert ALL columns to strings
            st.warning("⚠️ Data type issue detected. Displaying with simplified formatting.")
            df_all_str = df.copy()
            
            # Convert every single column to string, no exceptions
            for col in df_all_str.columns:
                try:
                    df_all_str[col] = df_all_str[col].fillna('').astype(str)
                except:
                    df_all_str[col] = [''] * len(df_all_str)
            
            # Try displaying again
            try:
                st.dataframe(df_all_str, **kwargs)
            except:
                # Last resort - display as text
                st.text("Data preview:")
                st.text(df_all_str.to_string())
        else:
            # Re-raise other errors
            raise e

# Override st.dataframe to always use safe display
_original_dataframe = st.dataframe

def safe_st_dataframe(data, **kwargs):
    """Override st.dataframe to always handle Arrow errors"""
    if isinstance(data, pd.DataFrame):
        return safe_dataframe_display(data, **kwargs)
    else:
        return _original_dataframe(data, **kwargs)

# Apply the override
st.dataframe = safe_st_dataframe

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
    
    /* AI suggestion box */
    .ai-suggestion {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
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

# Initialize AI clients
@st.cache_resource
def init_ai_clients():
    """Initialize AI clients if API keys are available"""
    clients = {'openai': None, 'anthropic': None}
    
    # Try OpenAI
    try:
        import openai
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            openai.api_key = st.secrets['OPENAI_API_KEY']
            clients['openai'] = openai
    except:
        pass
    
    # Try Anthropic
    try:
        import anthropic
        if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
            clients['anthropic'] = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])
    except:
        pass
    
    return clients

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
    filename = getattr(excel_file, 'filename', '')
    if filename:
        filename_lower = filename.lower()
        if 'top customer' in filename_lower or 'top volume' in filename_lower:
            return 'summary'
    
    # Check for sheets with embroid/imprint in name
    elif any(any(keyword in sheet.lower() for keyword in ['embroid', 'imprint', 'imp ']) 
             for sheet in sheet_names):
        return 'summary'
    
    # Default check - look at first sheet structure
    try:
        df = pd.read_excel(excel_file, sheet_name=0, nrows=5)
        # Check for customer/company columns
        cols_lower = [str(col).lower() for col in df.columns]
        if any('customer' in col or 'company' in col for col in cols_lower):
            return 'summary'
    except:
        pass
    
    return 'unknown'

def process_sales_analysis_format(excel_file):
    """Process the original Sales Analysis format with separate category sheets"""
    results = {}
    
    for sheet in excel_file.sheet_names:
        if 'Sales Analysis' in sheet:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            
            # Determine category from sheet name
            if 'Embroidery' in sheet:
                category = 'embroidery'
            elif 'Imprinting' in sheet:
                category = 'imprinting'
            else:
                category = sheet.replace('Sales Analysis', '').strip().lower()
            
            # Process the data
            if not df.empty:
                # Group by customer
                customer_summary = df.groupby('Sold To Customer Name').agg({
                    'Qty Delivered': 'sum',
                    'Item Number': lambda x: ', '.join(x.unique()[:5])  # Top 5 SKUs
                }).reset_index()
                
                customer_summary = customer_summary.rename(columns={
                    'Sold To Customer Name': 'Customer',
                    'Qty Delivered': 'Total Quantity',
                    'Item Number': 'Top SKUs'
                })
                
                # Sort by quantity
                customer_summary = customer_summary.sort_values('Total Quantity', ascending=False)
                
                results[category] = customer_summary
    
    return results

def process_summary_format(excel_file, file_content, use_ai=False):
    """Process the summary format with flexible sheet detection"""
    results = {}
    processing_stats = {
        'total_customers': 0,
        'total_quantity': 0,
        'unique_skus': set(),
        'sheets_processed': []
    }
    
    # Try multiple approaches to handle different formats
    approaches = [
        # Approach 1: Look for sheets with specific keywords
        lambda: process_by_sheet_names(excel_file, processing_stats),
        # Approach 2: Try to read all sheets and detect format
        lambda: process_all_sheets(excel_file, file_content, processing_stats, use_ai),
        # Approach 3: Process first two sheets as imp/embroid
        lambda: process_first_two_sheets(excel_file, processing_stats)
    ]
    
    for approach in approaches:
        try:
            results = approach()
            if results:
                break
        except Exception as e:
            continue
    
    return results, processing_stats

def process_by_sheet_names(excel_file, processing_stats):
    """Process sheets based on their names"""
    results = {}
    
    for sheet_name in excel_file.sheet_names:
        sheet_lower = sheet_name.lower()
        
        # Skip empty or summary sheets
        if any(skip in sheet_lower for skip in ['summary', 'total', 'overview']):
            continue
        
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            if df.empty or len(df) < 2:
                continue
            
            # Detect customer column
            customer_col = detect_customer_column(df)
            quantity_cols = detect_quantity_columns(df)
            
            if customer_col and quantity_cols:
                # Process the sheet
                result_df = process_sheet_data(df, customer_col, quantity_cols, processing_stats)
                
                # Determine category
                if any(keyword in sheet_lower for keyword in ['imp', 'imprint']):
                    results['imprinting'] = result_df
                elif any(keyword in sheet_lower for keyword in ['embroid']):
                    results['embroidery'] = result_df
                else:
                    # Use sheet name as category
                    category = sheet_name.replace(' ', '_').lower()
                    results[category] = result_df
                
                processing_stats['sheets_processed'].append(sheet_name)
        
        except Exception as e:
            continue
    
    return results

def process_all_sheets(excel_file, file_content, processing_stats, use_ai=False):
    """Process all sheets and try to detect format"""
    results = {}
    ai_clients = init_ai_clients() if use_ai else None
    
    # Try to read with different encoding if needed
    encodings = [None, 'utf-8', 'latin1', 'cp1252']
    
    sheets_processed = False
    
    for encoding in encodings:
        try:
            # Re-read the file with specific encoding
            if encoding:
                excel_file = pd.ExcelFile(io.BytesIO(file_content))
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    if df.empty or len(df) < 2:
                        continue
                    
                    # Use AI to help identify columns if enabled
                    if use_ai and ai_clients:
                        column_hints = get_ai_column_hints(df, ai_clients)
                    else:
                        column_hints = None
                    
                    # Detect columns
                    customer_col = detect_customer_column(df, column_hints)
                    quantity_cols = detect_quantity_columns(df, column_hints)
                    
                    if customer_col and quantity_cols:
                        result_df = process_sheet_data(df, customer_col, quantity_cols, processing_stats)
                        
                        # Try to determine category from sheet name or content
                        sheet_lower = sheet_name.lower()
                        if 'imp' in sheet_lower or 'imprint' in sheet_lower:
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
                
        except Exception as e:
            # Try next encoding
            continue
    
    # If no valid data found, try generic processing with AI
    if not results and excel_file.sheet_names:
        # Process first sheet as generic customer data
        results, processing_stats = process_summary_format(excel_file, file_content, use_ai)
    
    # Calculate average order size
    if processing_stats['total_customers'] > 0:
        processing_stats['avg_order_size'] = round(
            processing_stats['total_quantity'] / processing_stats['total_customers'], 1
        )
    
    # Validate and clean results before returning
    results = validate_results(results)
    
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
                sheet_info['has_data'] = len(df) > 0
                sheet_info['detected_columns'] = list(df.columns)
                
                if len(df) > 0:
                    # Try to detect customer column
                    customer_col = detect_customer_column(df)
                    if customer_col:
                        sheet_info['potential_customer_col'] = customer_col
                    
                    # Try to detect quantity columns
                    qty_cols = detect_quantity_columns(df)
                    if qty_cols:
                        sheet_info['potential_qty_col'] = qty_cols[0]
                
            except Exception as e:
                sheet_info['error'] = str(e)
            
            diagnostics['sheets'].append(sheet_info)
        
        # Analyze issues
        if not any(sheet['has_data'] for sheet in diagnostics['sheets']):
            diagnostics['issues'].append("No sheets contain data")
        
        if not any(sheet.get('potential_customer_col') for sheet in diagnostics['sheets']):
            diagnostics['issues'].append("No customer/company columns detected")
        
        # Provide recommendations
        if diagnostics['issues']:
            diagnostics['recommendations'].append(
                "Ensure your file has columns named 'Customer' or 'Company'"
            )
            diagnostics['recommendations'].append(
                "Check that data starts from row 1 with headers"
            )
    
    except Exception as e:
        diagnostics['file_readable'] = False
        diagnostics['issues'].append(f"Cannot read file: {str(e)}")
    
    return diagnostics

def process_first_two_sheets(excel_file, processing_stats):
    """Fallback: Process first two sheets as imp/embroid"""
    results = {}
    
    sheets = excel_file.sheet_names[:2]
    categories = ['imprinting', 'embroidery']
    
    for sheet_name, category in zip(sheets, categories):
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            if df.empty:
                continue
            
            customer_col = detect_customer_column(df)
            quantity_cols = detect_quantity_columns(df)
            
            if customer_col and quantity_cols:
                result_df = process_sheet_data(df, customer_col, quantity_cols, processing_stats)
                if category == 'embroidery':
                    result_df = result_df.rename(columns={'Customer': 'Company'})
                results[category] = result_df
                processing_stats['sheets_processed'].append(sheet_name)
        
        except:
            continue
    
    return results

def detect_customer_column(df, ai_hints=None):
    """Detect the customer/company column in the dataframe"""
    possible_names = ['customer', 'company', 'client', 'account', 'sold to', 'bill to', 
                      'customer name', 'company name', 'account name']
    
    # Check AI hints first
    if ai_hints and 'customer_column' in ai_hints:
        if ai_hints['customer_column'] in df.columns:
            return ai_hints['customer_column']
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        for name in possible_names:
            if name in col_lower:
                return col
    
    # Fallback: look for string columns with unique values
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_ratio = len(df[col].unique()) / len(df)
            if 0.1 < unique_ratio < 0.9:  # Reasonable number of unique values
                return col
    
    return None

def detect_quantity_columns(df, ai_hints=None):
    """Detect quantity/amount columns in the dataframe"""
    possible_names = ['quantity', 'qty', 'amount', 'total', 'units', 'pieces', 
                      'delivered', 'ordered', 'sales', 'volume']
    
    quantity_cols = []
    
    # Check AI hints first
    if ai_hints and 'quantity_columns' in ai_hints:
        for col in ai_hints['quantity_columns']:
            if col in df.columns:
                quantity_cols.append(col)
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        
        # Check name patterns
        for name in possible_names:
            if name in col_lower and col not in quantity_cols:
                # Verify it's numeric
                if df[col].dtype in ['int64', 'float64'] or can_convert_to_numeric(df[col]):
                    quantity_cols.append(col)
                    break
    
    # If no quantity columns found, look for numeric columns
    if not quantity_cols:
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if values are reasonable for quantities
                if df[col].min() >= 0 and df[col].max() < 1e9:
                    quantity_cols.append(col)
    
    return quantity_cols

def can_convert_to_numeric(series):
    """Check if a series can be converted to numeric"""
    try:
        pd.to_numeric(series, errors='coerce').notna().sum() > len(series) * 0.5
        return True
    except:
        return False

def process_sheet_data(df, customer_col, quantity_cols, processing_stats):
    """Process a single sheet's data"""
    # Clean the dataframe
    df = df.dropna(subset=[customer_col])
    
    # Create total quantity column
    if len(quantity_cols) == 1:
        total_col = quantity_cols[0]
    else:
        # Sum multiple quantity columns
        df['Total Quantity'] = df[quantity_cols].sum(axis=1)
        total_col = 'Total Quantity'
    
    # Ensure numeric
    df[total_col] = pd.to_numeric(df[total_col], errors='coerce').fillna(0)
    
    # Group by customer
    result = df.groupby(customer_col).agg({
        total_col: 'sum'
    }).reset_index()
    
    result = result.rename(columns={
        customer_col: 'Customer',
        total_col: 'Total Quantity'
    })
    
    # Add SKU information if available
    sku_cols = [col for col in df.columns if any(term in str(col).lower() 
                for term in ['sku', 'item', 'product', 'part'])]
    
    if sku_cols:
        # Get top SKUs per customer
        sku_info = df.groupby(customer_col)[sku_cols[0]].apply(
            lambda x: ', '.join(x.value_counts().head(5).index.astype(str))
        ).reset_index()
        sku_info = sku_info.rename(columns={sku_cols[0]: 'Top SKUs'})
        
        result = result.merge(sku_info, left_on='Customer', right_on=customer_col, how='left')
        
        # Update stats
        processing_stats['unique_skus'].update(df[sku_cols[0]].unique())
    
    # Sort by quantity
    result = result.sort_values('Total Quantity', ascending=False)
    
    # Update processing stats
    processing_stats['total_customers'] += len(result)
    processing_stats['total_quantity'] += result['Total Quantity'].sum()
    
    return result

def get_ai_column_hints(df, ai_clients):
    """Use AI to help identify columns"""
    if not ai_clients:
        return None
    
    # Prepare sample data for AI
    sample_data = {
        'columns': list(df.columns),
        'sample_rows': df.head(3).to_dict('records')
    }
    
    prompt = f"""Analyze this Excel data and identify:
1. Which column contains customer/company names
2. Which columns contain quantity/amount data

Data sample:
{json.dumps(sample_data, indent=2)}

Return as JSON:
{{"customer_column": "column_name", "quantity_columns": ["col1", "col2"]}}
"""
    
    # Try Claude first
    if ai_clients.get('anthropic'):
        try:
            response = ai_clients['anthropic'].messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            # Extract JSON from response
            text = response.content[0].text
            # Find JSON in response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            st.warning(f"Anthropic analysis failed: {e}")
    
    return None

def validate_results(results):
    """Validate and fix any remaining data type issues in results"""
    cleaned_results = {}
    
    for category, df in results.items():
        if df is not None and isinstance(df, pd.DataFrame):
            # Ensure all columns have proper types
            df_clean = fix_column_types(df.copy())
            
            # Double-check for any remaining object columns with mixed types
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    # Force to string to be safe
                    df_clean[col] = df_clean[col].fillna('').astype(str)
            
            cleaned_results[category] = df_clean
        else:
            cleaned_results[category] = df
    
    return cleaned_results

def create_visualizations(results):
    """Create visualizations for the analysis"""
    viz = {}
    
    for category, df in results.items():
        if df is not None and not df.empty:
            # Top 20 customers bar chart
            top_20 = df.head(20)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=top_20.index,
                y=top_20['Total Quantity'],
                text=top_20['Total Quantity'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                marker_color='#333'
            ))
            
            fig_bar.update_layout(
                title=f"Top 20 {category.title()} Customers",
                xaxis_title="Customer Rank",
                yaxis_title="Total Quantity",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12, color="#333"),
                height=400
            )
            
            # Pareto chart (80/20 analysis)
            df['Cumulative %'] = (df['Total Quantity'].cumsum() / df['Total Quantity'].sum() * 100)
            
            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(
                x=df.index[:20],
                y=df['Total Quantity'][:20],
                name='Quantity',
                marker_color='lightgray',
                yaxis='y'
            ))
            
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
                'Categories Analyzed'
            ],
            'Value': [
                stats.get('total_customers', 0),
                f"{stats.get('total_quantity', 0):,.0f}",
                len(stats.get('unique_skus', [])),
                f"{stats.get('avg_order_size', 0):,.1f}",
                ', '.join([cat.title() for cat in results.keys()])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        worksheet = writer.sheets['Summary']
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:B', 20)
        
        # Add header format
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Category sheets
        for category, df in results.items():
            if df is not None and not df.empty:
                # Add tier classification
                df['Tier'] = pd.cut(
                    df.index,
                    bins=[-1, len(df)*0.2, len(df)*0.5, len(df)],
                    labels=['A', 'B', 'C']
                )
                
                # Add percentage of total
                df['% of Total'] = (df['Total Quantity'] / df['Total Quantity'].sum() * 100).round(2)
                
                # Write to Excel
                sheet_name = category.title()[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                worksheet = writer.sheets[sheet_name]
                
                # Format columns
                worksheet.set_column('A:A', 40)  # Customer column
                worksheet.set_column('B:B', 15)  # Quantity
                if 'Top SKUs' in df.columns:
                    worksheet.set_column('C:C', 50)  # SKUs
                
                # Add header format
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Add conditional formatting for tiers
                tier_col = df.columns.get_loc('Tier')
                if tier_col >= 0:
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
    
    # AI toggle in sidebar
    with st.sidebar:
        st.markdown("### 🤖 AI Settings")
        use_ai = st.checkbox(
            "Enable AI Analysis",
            value=False,
            help="Use AI to help identify columns and categories in your file"
        )
        
        if use_ai:
            ai_clients = init_ai_clients()
            available_ai = []
            if ai_clients.get('openai'):
                available_ai.append("OpenAI ✅")
            if ai_clients.get('anthropic'):
                available_ai.append("Claude ✅")
            
            if available_ai:
                st.success(f"Available: {', '.join(available_ai)}")
            else:
                st.error("No AI APIs configured")
                st.info("Add OPENAI_API_KEY or ANTHROPIC_API_KEY to Streamlit secrets")
    
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
        
        # Show file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{file_size:.1f} MB")
        with col2:
            st.metric("File Type", uploaded_file.name.split('.')[-1].upper())
        with col3:
            upload_time = datetime.now().strftime("%H:%M:%S")
            st.metric("Uploaded At", upload_time)
        
        try:
            with st.spinner(f'Reading file... (Size: {file_size:.1f} MB)'):
                # Read file content once
                file_content = uploaded_file.read()
                
                # Create Excel file object
                excel_file = pd.ExcelFile(io.BytesIO(file_content))
                
                # Detect format
                file_format = detect_file_format(excel_file)
                
                st.info(f"📄 Detected format: **{file_format}**")
                
                # Show sheet information
                with st.expander("📋 File Structure", expanded=False):
                    for sheet in excel_file.sheet_names:
                        st.write(f"• {sheet}")
            
            # Process based on format
            with st.spinner('Processing data...'):
                start_time = time.time()
                
                if file_format == 'sales_analysis':
                    results = process_sales_analysis_format(excel_file)
                    processing_stats = {
                        'total_customers': sum(len(df) for df in results.values()),
                        'total_quantity': sum(df['Total Quantity'].sum() for df in results.values()),
                        'unique_skus': set()
                    }
                elif file_format in ['summary', 'unknown']:
                    results, processing_stats = process_summary_format(excel_file, file_content, use_ai)
                else:
                    st.error("Unable to process this file format")
                    
                    # Show diagnostic information
                    with st.expander("🔍 Diagnostic Information", expanded=True):
                        diagnostics = validate_and_diagnose_file(file_content)
                        
                        st.write("**File Status:**")
                        st.write(f"- Readable: {'✅' if diagnostics['file_readable'] else '❌'}")
                        st.write(f"- Sheets found: {len(diagnostics['sheets'])}")
                        
                        if diagnostics['issues']:
                            st.write("**Issues:**")
                            for issue in diagnostics['issues']:
                                st.write(f"- ❗ {issue}")
                        
                        if diagnostics['recommendations']:
                            st.write("**Recommendations:**")
                            for rec in diagnostics['recommendations']:
                                st.write(f"- 💡 {rec}")
                        
                        if diagnostics['sheets']:
                            st.write("**Sheet Details:**")
                            for sheet in diagnostics['sheets']:
                                st.write(f"\n**{sheet['name']}:**")
                                st.write(f"  - Rows: {sheet['rows']}")
                                st.write(f"  - Columns: {sheet['cols']}")
                                if sheet.get('potential_customer_col'):
                                    st.write(f"  - Customer column: {sheet['potential_customer_col']}")
                                if sheet.get('potential_qty_col'):
                                    st.write(f"  - Quantity column: {sheet['potential_qty_col']}")
                    return
                
                processing_time = time.time() - start_time
            
            # Display results
            if results:
                # Summary metrics
                st.markdown("### 📊 Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Customers", f"{processing_stats.get('total_customers', 0):,}")
                with col2:
                    st.metric("Total Quantity", f"{processing_stats.get('total_quantity', 0):,.0f}")
                with col3:
                    st.metric("Unique SKUs", f"{len(processing_stats.get('unique_skus', [])):,}")
                with col4:
                    st.metric("Processing Time", f"{processing_time:.1f}s")
                
                # Category tabs
                if len(results) > 1:
                    tabs = st.tabs([cat.title() for cat in results.keys()])
                    
                    for idx, (category, df) in enumerate(results.items()):
                        with tabs[idx]:
                            display_category_results(category, df)
                else:
                    # Single category
                    category, df = next(iter(results.items()))
                    display_category_results(category, df)
                
                # Visualizations
                st.markdown("### 📈 Visual Analysis")
                viz = create_visualizations(results)
                
                # Display charts in columns
                for category in results.keys():
                    if f'{category}_bar' in viz and f'{category}_pareto' in viz:
                        st.markdown(f"#### {category.title()} Analysis")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(viz[f'{category}_bar'], use_container_width=True)
                        with col2:
                            st.plotly_chart(viz[f'{category}_pareto'], use_container_width=True)
                
                # Download section
                st.markdown("### 💾 Export Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Excel download
                    excel_output = create_excel_output(results, processing_stats)
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_output,
                        file_name=f"embroid_imprint_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # CSV download (combined)
                    combined_df = pd.DataFrame()
                    for category, df in results.items():
                        df_copy = df.copy()
                        df_copy['Category'] = category.title()
                        combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
                    
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV (Combined)",
                        data=csv,
                        file_name=f"embroid_imprint_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            else:
                st.error("❌ No data could be extracted from the file")
                st.info("Please ensure your file contains customer and quantity data")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
            # Detailed error info in expander
            with st.expander("🔍 Error Details", expanded=True):
                st.code(str(e))
                st.write("**Suggestions:**")
                st.write("1. Ensure the file is a valid Excel file (.xlsx or .xls)")
                st.write("2. Check that the file is not password protected")
                st.write("3. Verify the file contains data in a tabular format")
                st.write("4. Try saving the file in a different Excel format")
                
                if use_ai:
                    st.write("5. AI analysis is enabled - check your API keys are valid")
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("### 📤 How to use this tool:")
        
        with st.expander("📋 Supported File Formats", expanded=True):
            st.markdown("""
            **Format 1: Sales Analysis Format**
            - Multiple sheets named "Sales Analysis - Embroidery", "Sales Analysis - Imprinting", etc.
            - Columns: Sold To Customer Name, Qty Delivered, Item Number
            
            **Format 2: Customer Summary Format**
            - Sheets with customer data (flexible naming)
            - Must have customer/company column and quantity columns
            - Tool will auto-detect columns with AI assistance (if enabled)
            """)
        
        with st.expander("🎯 Expected Output", expanded=True):
            st.markdown("""
            - **Customer Rankings**: Top customers by volume for each category
            - **Pareto Analysis**: 80/20 breakdown showing customer concentration
            - **SKU Information**: Top products per customer (if available)
            - **Tier Classification**: A/B/C customer segmentation
            - **Excel Report**: Multi-sheet workbook with detailed analysis
            """)
        
        with st.expander("💡 Tips for Best Results", expanded=True):
            st.markdown("""
            - Ensure customer names are consistent (no duplicates with slight variations)
            - Include quantity/volume data in numeric format
            - Remove any summary rows or subtotals before uploading
            - For large files (>50MB), consider splitting by time period
            - Enable AI analysis for better column detection
            """)
        
        # Sample file download
        with st.expander("📥 Download Sample File", expanded=False):
            st.markdown("Download a sample file to see the expected format:")
            
            # Create sample data
            sample_data_imp = {
                'Customer': ['ABC Company', 'XYZ Corp', 'Tech Solutions Inc', 'Global Traders', 'Smart Systems'],
                'Total Quantity': [15000, 12000, 8000, 6000, 4000],
                'Top SKUs': ['SKU-001, SKU-002', 'SKU-003', 'SKU-001, SKU-004', 'SKU-002', 'SKU-005']
            }
            
            sample_data_emb = {
                'Company': ['Fashion House', 'Sports Gear Ltd', 'Uniform Specialists', 'Promo Products', 'Event Planners'],
                'Total Quantity': [20000, 18000, 15000, 10000, 7000],
                'Top SKUs': ['EMB-101, EMB-102', 'EMB-201', 'EMB-101, EMB-301', 'EMB-401', 'EMB-501']
            }
            
            with io.BytesIO() as output:
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    pd.DataFrame(sample_data_imp).to_excel(writer, sheet_name='Imp Top Customers', index=False)
                    pd.DataFrame(sample_data_emb).to_excel(writer, sheet_name='Embroid Top Customers', index=False)
                
                output.seek(0)
                st.download_button(
                    label="📥 Download Sample File",
                    data=output,
                    file_name="Imp_Embroid_Top_Customers_Sample.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

def display_category_results(category, df):
    """Display results for a single category"""
    st.markdown(f"### {category.title()} Top Customers")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Total Volume", f"{df['Total Quantity'].sum():,.0f}")
    with col3:
        top_10_pct = (df.head(10)['Total Quantity'].sum() / df['Total Quantity'].sum() * 100)
        st.metric("Top 10 Share", f"{top_10_pct:.1f}%")
    
    # Customer table
    st.markdown("#### Top 20 Customers")
    
    # Prepare display dataframe
    display_df = df.head(20).copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = 'Rank'
    
    # Format numbers
    display_df['Total Quantity'] = display_df['Total Quantity'].apply(lambda x: f'{x:,.0f}')
    
    # Display with custom styling
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # 80/20 Analysis
    total_customers = len(df)
    customers_for_80 = len(df[df['Cumulative %'] <= 80]) if 'Cumulative %' in df.columns else 0
    
    if customers_for_80 > 0:
        st.info(f"💡 **80/20 Analysis**: {customers_for_80} customers ({customers_for_80/total_customers*100:.1f}%) account for 80% of volume")

if __name__ == "__main__":
    main()
