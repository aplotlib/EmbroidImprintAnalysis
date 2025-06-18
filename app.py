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

@st.cache_data
def process_customer_sku_data(file_content, filename):
    """
    Process Excel file with caching for better performance.
    """
    # Read Excel file from bytes
    excel_file = pd.ExcelFile(io.BytesIO(file_content))
    
    results = {}
    processing_stats = {
        'total_customers': 0,
        'total_skus': set(),
        'total_quantity': 0,
        'avg_order_size': 0
    }
    
    for sheet_type in ['Imprinting - Sales Analysis', 'Embroidery - Sales Analysis']:
        if sheet_type in excel_file.sheet_names:
            # Read with optimization for large files
            df = pd.read_excel(
                io.BytesIO(file_content), 
                sheet_name=sheet_type,
                engine='openpyxl'
            )
            
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
                            qty = float(row.iloc[7]) if pd.notna(row.iloc[7]) else 0
                        except:
                            qty = 0
                        
                        if qty > 0:
                            customer_data[customer_name]['total_qty'] += qty
                            customer_data[customer_name]['skus'].add(current_sku)
                            customer_data[customer_name]['order_count'] += 1
                            processing_stats['total_quantity'] += qty
            
            # Convert to DataFrame with additional analytics
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
            if 'Imprinting' in sheet_type:
                results['imprinting'] = result_df
            else:
                result_df = result_df.rename(columns={'Customer': 'Company'})
                results['embroidery'] = result_df
    
    # Calculate average order size
    if processing_stats['total_customers'] > 0:
        processing_stats['avg_order_size'] = round(
            processing_stats['total_quantity'] / processing_stats['total_customers'], 1
        )
    
    # Clean up memory for large files
    gc.collect()
    
    return results, processing_stats

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
        help="Supports large files up to 200MB"
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
                st.error("No valid data sheets found in the file.")
                return
            
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
            st.markdown("""
            <div class="info-box">
            <strong>Troubleshooting:</strong><br>
            • Ensure the file contains 'Imprinting - Sales Analysis' or 'Embroidery - Sales Analysis' sheets<br>
            • Check that data starts from row 5 with customer names in column A<br>
            • Verify quantities are in column H<br>
            • Try saving the file in .xlsx format if using .xls
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        ### 📁 How to use:
        
        1. **Upload** your Excel file containing sales analysis data
        2. **View** automated analysis with customer rankings and SKU patterns  
        3. **Export** comprehensive reports for further analysis
        
        ### 📊 What you'll get:
        
        - **Customer Rankings** - Identify your top volume customers
        - **Pareto Analysis** - See which customers drive 80% of your business
        - **SKU Insights** - Understand product ordering patterns
        - **Revenue Tiers** - Automatic A/B/C customer classification
        
        ### ⚡ Features:
        
        - Handles files up to 200MB
        - Processes multiple sheets automatically
        - Real-time search and filtering
        - Export to Excel with formatting
        """)
        
        # Sample data structure
        with st.expander("📋 Expected File Format"):
            st.markdown("""
            Your Excel file should contain sheets named:
            - `Imprinting - Sales Analysis`
            - `Embroidery - Sales Analysis`
            
            Each sheet should have:
            - Customer names in column A
            - SKU information in brackets [SKU123] 
            - Quantities in column H
            - Data starting from row 5
            """)

if __name__ == "__main__":
    main()
