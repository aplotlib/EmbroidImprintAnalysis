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
                # Extract SKU code
                sku_match = re.search(r'\[([^\]]+)\]', first_cell)
                if sku_match:
                    current_sku = sku_match.group(1)
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
    st.markdown(f"### {category.title()} Top Customers")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Total Volume", f"{df['Total Quantity'].sum():,.0f}")
    with col3:
        top_10_pct = (df.head(10)['Total Quantity'].sum() / df['Total Quantity'].sum() * 100)
        st.metric("Top 10 Share", f"{top_10_pct:.1f}%")
    with col4:
        if 'SKU Count' in df.columns:
            st.metric("Unique SKUs", df['SKU Count'].sum())
    
    # Top 10 Customers Highlight
    st.markdown("#### 🏆 Top 10 Customers")
    top_10_df = df.head(10).copy()
    
    # Format for display
    display_cols = ['Rank']
    customer_col = 'Company' if 'Company' in top_10_df.columns else 'Customer'
    display_cols.append(customer_col)
    display_cols.extend(['Total Quantity', 'SKUs', 'SKU Count'])
    
    # Select only available columns
    display_cols = [col for col in display_cols if col in top_10_df.columns]
    top_10_display = top_10_df[display_cols].copy()
    
    # Format numbers
    top_10_display['Total Quantity'] = top_10_display['Total Quantity'].apply(lambda x: f'{x:,.0f}')
    
    # Display with highlighting
    st.dataframe(
        top_10_display,
        use_container_width=True,
        height=400,
        hide_index=True
    )
    
    # All customers table
    with st.expander("📊 View All Customers", expanded=False):
        st.markdown("#### Complete Customer List")
        
        # Prepare display dataframe
        all_display = df[display_cols].copy()
        all_display['Total Quantity'] = all_display['Total Quantity'].apply(lambda x: f'{x:,.0f}')
        
        st.dataframe(
            all_display,
            use_container_width=True,
            height=600,
            hide_index=True
        )
    
    # 80/20 Analysis
    total_customers = len(df)
    customers_for_80 = len(df[df['Cumulative %'] <= 80]) if 'Cumulative %' in df.columns else 0
    
    if customers_for_80 > 0:
        st.info(f"💡 **80/20 Analysis**: {customers_for_80} customers ({customers_for_80/total_customers*100:.1f}%) account for 80% of volume")
    
    # SKU Analysis if available
    if 'SKUs' in df.columns and df['SKUs'].notna().any():
        with st.expander("📦 SKU Analysis", expanded=False):
            # Extract all SKUs
            all_skus = []
            for sku_list in df['SKUs'].dropna():
                if sku_list:
                    all_skus.extend([sku.strip() for sku in sku_list.split(',')])
            
            if all_skus:
                sku_counts = pd.Series(all_skus).value_counts().head(10)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Most Popular SKUs:**")
                    for i, (sku, count) in enumerate(sku_counts.items(), 1):
                        st.write(f"{i}. {sku}: {count} customers")
                
                with col2:
                    # Create a simple bar chart for SKUs
                    fig_sku = go.Figure(data=[
                        go.Bar(
                            x=sku_counts.values,
                            y=sku_counts.index,
                            orientation='h',
                            marker_color='#333'
                        )
                    ])
                    fig_sku.update_layout(
                        title="Top SKUs by Customer Count",
                        xaxis_title="Number of Customers",
                        yaxis_title="SKU",
                        height=300,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_sku, use_container_width=True)

def main():
    # Header
    st.markdown("# 🎨 Embroid/Imprint Analysis")
    st.markdown("Upload sales analysis files to identify top customers and SKU patterns")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Select Excel file",
        type=['xlsx', 'xls'],
        help="Supports hierarchical format with SKUs and customer data"
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
                # Read file content
                file_content = uploaded_file.read()
                
            # Process the file
            with st.spinner('Processing data...'):
                start_time = time.time()
                results, processing_stats = process_excel_file(file_content)
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
                        # Standardize column names
                        if 'Company' in df_copy.columns:
                            df_copy = df_copy.rename(columns={'Company': 'Customer'})
                        combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
                    
                    if not combined_df.empty:
                        csv = combined_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV (Combined)",
                            data=csv,
                            file_name=f"embroid_imprint_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            else:
                st.error("❌ No data could be extracted from the file")
                
                # Show what was detected
                with st.expander("🔍 File Analysis", expanded=True):
                    try:
                        excel_file = pd.ExcelFile(io.BytesIO(file_content))
                        st.write(f"**Sheets found:** {excel_file.sheet_names}")
                        
                        st.write("\n**Expected format:**")
                        st.markdown("""
                        - Sheet names should contain 'Imprint' or 'Embroid'
                        - SKUs should be in brackets like `[RHB1068GRYIMP]`
                        - Customer names should be indented below SKUs
                        - Last column should contain totals
                        """)
                        
                        # Show sample of first sheet
                        if excel_file.sheet_names:
                            df_sample = pd.read_excel(excel_file, sheet_name=0, nrows=10)
                            st.write("\n**First 10 rows of first sheet:**")
                            st.dataframe(df_sample)
                    except:
                        pass
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
            # Detailed error info
            with st.expander("🔍 Error Details & Troubleshooting", expanded=True):
                st.code(str(e))
                
                st.write("**Common issues:**")
                st.write("1. Make sure your file has the hierarchical format (SKUs with brackets, indented customers)")
                st.write("2. Sheet names should contain 'Imprint' or 'Embroid'")
                st.write("3. Ensure the file is not corrupted or password protected")
                st.write("4. Check that totals are in the last column")
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("### 📤 How to use this tool:")
        
        with st.expander("📋 Expected File Format", expanded=True):
            st.markdown("""
            **Hierarchical Format Requirements:**
            - SKUs are shown with brackets like `[RHB1068GRYIMP] Product Name`
            - Customer names are indented underneath each SKU
            - Monthly data in columns with dates as headers
            - Total quantity in the last column
            - Sheet names should contain 'Imprint' or 'Embroid'
            
            **Example Structure:**
            ```
            |                                          | Aug 2024 | Sep 2024 | ... | Total |
            |------------------------------------------|----------|----------|-----|-------|
            | Total                                    | 100      | 150      | ... | 1000  |
            | [SKU123] Product Name                    |          |          |     | 50    |
            |     Customer ABC Company                 | 10       | 15       | ... | 25    |
            |     Customer XYZ Corp                    | 5        | 20       | ... | 25    |
            | [SKU456] Another Product                 |          |          |     | 100   |
            |     Customer 123 Inc                     | 30       | 40       | ... | 70    |
            |     Customer DEF Ltd                     | 20       | 10       | ... | 30    |
            ```
            """)
        
        with st.expander("🎯 What This Tool Does", expanded=True):
            st.markdown("""
            - **Aggregates customer totals** across all SKUs they've purchased
            - **Identifies top 10 customers** with special highlighting
            - **Performs 80/20 analysis** to show customer concentration
            - **Tracks SKU diversity** per customer
            - **Exports complete data** with rankings and tier classifications
            - **Handles large files** up to 10,000 rows per sheet
            """)
        
        with st.expander("💡 Tips for Best Results", expanded=True):
            st.markdown("""
            - Ensure customer names are consistent (avoid variations of the same company)
            - The total column should be the rightmost column
            - Remove any manual subtotals or summary rows
            - Keep the hierarchical structure intact (SKUs → Customers)
            - For files larger than 200MB, consider splitting by date range
            """)

if __name__ == "__main__":
    main()
