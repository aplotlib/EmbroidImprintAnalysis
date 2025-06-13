import pandas as pd
import streamlit as st
import io
from collections import defaultdict

def process_customer_sku_data(uploaded_file):
    """
    Process the Excel file to extract unique customers with their total quantities and SKUs.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        dict: Dictionary with 'imprinting' and 'embroidery' DataFrames
    """
    # Read the Excel file
    excel_file = pd.ExcelFile(uploaded_file)
    
    # Process both sheets
    results = {}
    
    for sheet_type in ['Imprinting - Sales Analysis', 'Embroidery - Sales Analysis']:
        if sheet_type in excel_file.sheet_names:
            # Read the sheet
            df = pd.read_excel(uploaded_file, sheet_name=sheet_type)
            
            # Initialize customer data dictionary
            customer_data = defaultdict(lambda: {'total_qty': 0, 'skus': set()})
            current_sku = None
            
            # Process each row
            for idx, row in df.iterrows():
                # Skip first few rows (headers and totals)
                if idx < 4:
                    continue
                
                # Get the value from column A (customer/SKU)
                col_a_value = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
                
                # Check if it's a SKU row (contains brackets)
                if '[' in col_a_value and ']' in col_a_value:
                    # Extract SKU code from the brackets
                    sku_start = col_a_value.find('[')
                    sku_end = col_a_value.find(']')
                    current_sku = col_a_value[sku_start:sku_end+1] + col_a_value[sku_end+1:].strip()
                else:
                    # It's a customer row
                    customer_name = col_a_value.strip()
                    
                    # Skip if it's empty or "Total"
                    if customer_name and customer_name != 'Total' and current_sku:
                        # Get quantity from column H (index 7)
                        try:
                            qty = float(row.iloc[7]) if pd.notna(row.iloc[7]) else 0
                        except:
                            qty = 0
                        
                        # Add to customer data
                        customer_data[customer_name]['total_qty'] += qty
                        customer_data[customer_name]['skus'].add(current_sku)
            
            # Convert to DataFrame
            data_list = []
            for customer, data in customer_data.items():
                data_list.append({
                    'Customer': customer,
                    'Qty Delivered': int(data['total_qty']),
                    'SKUs Ordered': ', '.join(sorted(data['skus']))
                })
            
            # Create DataFrame and sort by quantity descending
            result_df = pd.DataFrame(data_list)
            result_df = result_df.sort_values('Qty Delivered', ascending=False)
            result_df = result_df.reset_index(drop=True)
            
            # Store in results
            if 'Imprinting' in sheet_type:
                results['imprinting'] = result_df
            else:
                # For embroidery sheet, rename 'Customer' to 'Company' to match expected format
                result_df = result_df.rename(columns={'Customer': 'Company'})
                results['embroidery'] = result_df
    
    return results

def create_excel_output(results):
    """
    Create an Excel file with the processed data.
    
    Args:
        results: Dictionary with 'imprinting' and 'embroidery' DataFrames
        
    Returns:
        BytesIO: Excel file in memory
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write Imprinting sheet
        if 'imprinting' in results:
            results['imprinting'].to_excel(
                writer, 
                sheet_name='Imprinting Top Customers', 
                index=False
            )
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Imprinting Top Customers']
            for i, col in enumerate(results['imprinting'].columns):
                if col == 'SKUs Ordered':
                    worksheet.set_column(i, i, 80)  # Wide column for SKUs
                elif col == 'Customer':
                    worksheet.set_column(i, i, 40)  # Medium width for customer names
                else:
                    worksheet.set_column(i, i, 15)  # Standard width
        
        # Write Embroidery sheet
        if 'embroidery' in results:
            results['embroidery'].to_excel(
                writer, 
                sheet_name='Embroidery Top Customers', 
                index=False
            )
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Embroidery Top Customers']
            for i, col in enumerate(results['embroidery'].columns):
                if col == 'SKUs Ordered':
                    worksheet.set_column(i, i, 80)  # Wide column for SKUs
                elif col == 'Company':
                    worksheet.set_column(i, i, 40)  # Medium width for company names
                else:
                    worksheet.set_column(i, i, 15)  # Standard width
    
    output.seek(0)
    return output

# Streamlit app
def main():
    st.title("Customer SKU Processor")
    st.write("Upload your Excel file to process customer data with SKU tracking")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel file", 
        type=['xlsx', 'xls'],
        help="Upload the 'Imprint_Embroid top volume customers' file"
    )
    
    if uploaded_file is not None:
        try:
            # Process the file
            with st.spinner('Processing file...'):
                results = process_customer_sku_data(uploaded_file)
            
            # Display summary
            st.success("File processed successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imprinting Summary")
                if 'imprinting' in results:
                    st.metric("Total Customers", len(results['imprinting']))
                    st.metric("Total Quantity", f"{results['imprinting']['Qty Delivered'].sum():,}")
                    
                    st.write("Top 5 Customers:")
                    st.dataframe(
                        results['imprinting'].head(5)[['Customer', 'Qty Delivered']], 
                        hide_index=True
                    )
            
            with col2:
                st.subheader("Embroidery Summary")
                if 'embroidery' in results:
                    st.metric("Total Customers", len(results['embroidery']))
                    st.metric("Total Quantity", f"{results['embroidery']['Qty Delivered'].sum():,}")
                    
                    st.write("Top 5 Companies:")
                    st.dataframe(
                        results['embroidery'].head(5)[['Company', 'Qty Delivered']], 
                        hide_index=True
                    )
            
            # Create download button
            excel_output = create_excel_output(results)
            
            st.download_button(
                label="Download Processed Excel File",
                data=excel_output,
                file_name="Customer_SKU_Summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Optional: Show full data tables
            with st.expander("View Full Imprinting Data"):
                if 'imprinting' in results:
                    st.dataframe(results['imprinting'], hide_index=True, height=400)
            
            with st.expander("View Full Embroidery Data"):
                if 'embroidery' in results:
                    st.dataframe(results['embroidery'], hide_index=True, height=400)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure you've uploaded the correct file format.")

if __name__ == "__main__":
    main()
