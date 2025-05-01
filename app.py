import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
from datetime import datetime
import logging
import xlsxwriter
import tempfile
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Set up logging (reduced verbosity)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit app title
st.title("Futuristic Invoice Data Analyzer")

# Cache filter options for performance
@st.cache_data
def get_filter_options(df, column):
    return sorted([str(x) for x in df[column].dropna().unique()])

# Cache data loading
@st.cache_data
def load_data(uploaded_file):
    try:
        tmp_file = None
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx' if uploaded_file.name.endswith('.xlsx') else '.csv') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_file = tmp.name
        
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(tmp_file, engine='openpyxl', dtype_backend='numpy_nullable')
        else:
            df_chunks = pd.read_csv(tmp_file, chunksize=10000, dtype_backend='numpy_nullable')
            df = pd.concat([chunk for chunk in df_chunks], ignore_index=True)
        
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)  # Clean up temporary file
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        logger.error(f"File loading error: {e}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload your .xlsx or .csv file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    st.success(f"Loaded {len(df)} rows from {uploaded_file.name}")

    # Validate required columns
    required_cols = ['Invoice No', 'Quantity', 'Rate', 'Discount', 'Tax Percentage', 'Tax Amount', 'Amount', 'Final Amount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}. Please ensure all required columns are present.")
        logger.error(f"Missing columns: {missing_cols}")
        st.stop()

    # Validate Invoice No
    if df['Invoice No'].dropna().empty:
        st.error("The 'Invoice No' column is entirely empty or null. Please ensure it contains valid values.")
        logger.error("Invoice No column is empty or all null.")
        st.stop()

    # Data cleaning options
    with st.expander("Data Cleaning", expanded=False):
        fill_na = st.checkbox("Fill missing numeric values with 0", value=True)
        if fill_na:
            numeric_cols = ['Quantity', 'Rate', 'Discount', 'Tax Percentage', 'Tax Amount', 'Amount', 'Final Amount']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert Bill Date to datetime
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
        except:
            return pd.NaT

    if 'Bill Date' in df.columns:
        df['Bill Date'] = df['Bill Date'].apply(parse_date)
        if df['Bill Date'].isna().all():
            st.warning("All Bill Date values are invalid. Date filtering disabled.")
            logger.warning("All Bill Date values are invalid.")

    # Add Tax Type column
    def assign_tax_type(row):
        try:
            tax_pct = float(row['Tax Percentage'])
            if row.get('Source of Supply') == 'MH':
                cgst = tax_pct / 2
                return f"CGST: {cgst:.1f}%, SGST: {cgst:.1f}%"
            return f"IGST: {tax_pct:.1f}%"
        except:
            return "Unknown"

    if 'Source of Supply' in df.columns and 'Tax Percentage' in df.columns:
        df['Tax Type'] = df.apply(assign_tax_type, axis=1)

    # Validate Calculations
    @st.cache_data
    def validate_calculations(df):
        try:
            numeric_cols = ['Quantity', 'Rate', 'Discount', 'Tax Percentage', 'Tax Amount', 'Amount', 'Final Amount']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['Base Amount'] = df['Quantity'] * df['Rate'] * (1 - df['Discount'] / 100)
            df['Calculated Tax Percentage'] = df.apply(
                lambda row: (row['Tax Amount'] / row['Base Amount'] * 100) if row['Base Amount'] != 0 else 0, axis=1
            )
            df['Tax Percentage Error'] = df['Calculated Tax Percentage'] - df['Tax Percentage']
            df['Tax Percentage Error'] = df['Tax Percentage Error'].fillna(0)
            
            standard_rates = [0, 5, 12, 18, 28]
            df['Tax Percentage Valid'] = df.apply(
                lambda row: 'Valid' if (row['Tax Percentage'] in standard_rates and 
                                       (pd.isna(row['Tax Percentage Error']) or abs(row['Tax Percentage Error']) < 0.1)) 
                                       else 'Invalid', axis=1
            )
            
            df['Calculated Quantity'] = df.apply(
                lambda row: (row['Amount'] / (row['Rate'] * (1 - row['Discount'] / 100))) 
                            if row['Rate'] != 0 and (1 - row['Discount'] / 100) != 0 else pd.NA, axis=1
            )
            df['Quantity Error'] = df.apply(
                lambda row: abs(row['Calculated Quantity'] - row['Quantity']) if pd.notna(row['Calculated Quantity']) else pd.NA, axis=1
            )
            df['Quantity Valid'] = df['Quantity Error'].apply(
                lambda x: 'Valid' if pd.isna(x) or x < 0.01 else 'Invalid'
            )
            
            df['Calculated Final Amount'] = df['Amount'] + df['Tax Amount']
            df['Final Amount Error'] = df['Calculated Final Amount'] - df['Final Amount']
            df['Final Amount Valid'] = df['Final Amount Error'].apply(
                lambda x: 'Valid' if pd.isna(x) or abs(x) < 0.01 else 'Invalid'
            )
            
            df['Final Amount Matches'] = df.apply(
                lambda row: 'Yes' if row['Tax Percentage Valid'] == 'Invalid' and row['Final Amount Valid'] == 'Valid'
                            else 'No' if row['Tax Percentage Valid'] == 'Invalid' and row['Final Amount Valid'] == 'Invalid'
                            else 'N/A', axis=1
            )
            
            df['Calculation Details'] = df.apply(
                lambda row: (
                    f"Base Amount: {row['Quantity']} * {row['Rate']} * (1 - {row['Discount']}/100) = {row['Base Amount']:.2f}\n"
                    f"Calculated Tax Percentage: {row['Tax Amount']:.2f} / {row['Base Amount']:.2f} * 100 = {row['Calculated Tax Percentage']:.2f}%\n"
                    f"Calculated Quantity: {row['Amount']:.2f} / ({row['Rate']:.2f} * (1 - {row['Discount']}/100)) = "
                    f"{row['Calculated Quantity']:.2f}" if row['Rate'] != 0 and (1 - row['Discount'] / 100) != 0 else "Invalid (Rate or Discount issue)\n"
                    f"Calculated Final Amount: {row['Amount']:.2f} + {row['Tax Amount']:.2f} = {row['Calculated Final Amount']:.2f}"
                ), axis=1
            )
            
            return df
        except Exception as e:
            st.error(f"Error in calculations validation: {e}")
            logger.error(f"Validation error: {e}")
            return df

    df = validate_calculations(df)

    # Sidebar for filters
    st.sidebar.header("Filters")
    filter_debug = []
    active_filters = []
    
    # Initialize session state for filter persistence
    if 'filter_state' not in st.session_state:
        st.session_state.filter_state = {}

    # Clear all filters button
    if st.sidebar.button("Clear All Filters"):
        st.session_state.clear()
        st.experimental_rerun()

    # Save filter state button
    if st.sidebar.button("Save Filter State"):
        st.session_state.filter_state = active_filters
        st.success("Filter state saved!")

    # Load saved filter state
    if st.sidebar.button("Load Saved Filter State"):
        if st.session_state.filter_state:
            st.success("Loaded saved filter state!")
        else:
            st.warning("No saved filter state found.")

    # Display filtered rows count
    st.sidebar.write(f"Filtered Rows: {len(df)} (before filters)")
    
    filtered_df = df.copy()

    # Date Filters
    with st.sidebar.expander("Date Filters", expanded=True):
        if 'Bill Date' in df.columns and not df['Bill Date'].isna().all():
            min_date = df['Bill Date'].min()
            max_date = df['Bill Date'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                date_range = st.date_input("Bill Date Range", [min_date, max_date], 
                                          min_value=min_date, max_value=max_date,
                                          help="Select a date range for Bill Date.", key="date_range")
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    prev_len = len(filtered_df)
                    filtered_df = filtered_df[(filtered_df['Bill Date'] >= pd.Timestamp(start_date)) & 
                                             (filtered_df['Bill Date'] <= pd.Timestamp(end_date)) | 
                                             filtered_df['Bill Date'].isna()]
                    filter_debug.append(f"Bill Date Range filter: {prev_len} -> {len(filtered_df)} rows")
                    active_filters.append(f"Bill Date: {start_date} to {end_date}")
                    logger.info(f"Bill Date Range filter applied: {len(filtered_df)} rows remain")
                
                years = sorted([y for y in filtered_df['Bill Date'].dt.year.unique() if not pd.isna(y)])
                if years:
                    select_all_years = st.checkbox("Select All Years", value=True, key="select_all_years")
                    selected_years = st.multiselect("Select Years", years, default=years if select_all_years else [],
                                                   help="Filter by specific years.", key="years")
                    if selected_years:
                        prev_len = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['Bill Date'].dt.year.isin(selected_years) | filtered_df['Bill Date'].isna()]
                        filter_debug.append(f"Years filter: {prev_len} -> {len(filtered_df)} rows")
                        active_filters.append(f"Years: {len(selected_years)} selected")
                        logger.info(f"Years filter applied: {len(filtered_df)} rows remain")
                
                months = sorted([m for m in filtered_df['Bill Date'].dt.month_name().unique() if not pd.isna(m)])
                if months:
                    select_all_months = st.checkbox("Select All Months", value=True, key="select_all_months")
                    selected_months = st.multiselect("Select Months", months, default=months if select_all_months else [],
                                                    help="Filter by specific months.", key="months")
                    if selected_months:
                        prev_len = len(filtered_df)
                        filtered_df = filtered_df[filtered_df['Bill Date'].dt.month_name().isin(selected_months) | filtered_df['Bill Date'].isna()]
                        filter_debug.append(f"Months filter: {prev_len} -> {len(filtered_df)} rows")
                        active_filters.append(f"Months: {len(selected_months)} selected")
                        logger.info(f"Months filter applied: {len(filtered_df)} rows remain")
        else:
            filter_debug.append("Bill Date filter: Skipped (all invalid or missing)")
            logger.info("Bill Date filter skipped: All invalid or missing")

    # Numeric Filters
    with st.sidebar.expander("Numeric Filters", expanded=True):
        numeric_cols = ['Quantity', 'Rate', 'Discount', 'Tax Percentage', 'Tax Amount', 'Amount', 'Final Amount', 
                        'Base Amount', 'Calculated Tax Percentage', 'Tax Percentage Error', 'Quantity Error', 'Final Amount Error']
        for col in numeric_cols:
            if col in filtered_df.columns:
                numeric_series = pd.to_numeric(filtered_df[col], errors='coerce')
                if numeric_series.notna().any():
                    min_val = numeric_series.min()
                    max_val = numeric_series.max()
                    if not pd.isna(min_val) and not pd.isna(max_val):
                        try:
                            selected_range = st.slider(f"{col}", float(min_val), float(max_val), (float(min_val), float(max_val)),
                                                      help=f"Filter {col} between {min_val:.2f} and {max_val:.2f}", key=f"slider_{col}")
                            prev_len = len(filtered_df)
                            filtered_df = filtered_df[(numeric_series >= selected_range[0]) & (numeric_series <= selected_range[1]) | numeric_series.isna()]
                            filter_debug.append(f"{col} filter: {prev_len} -> {len(filtered_df)} rows")
                            active_filters.append(f"{col}: {selected_range[0]:.2f} to {selected_range[1]:.2f}")
                            logger.info(f"{col} filter applied: {len(filtered_df)} rows remain")
                        except Exception as e:
                            st.warning(f"Slider error for {col}: {e}")
                            logger.error(f"Slider error for {col}: {e}")
                    else:
                        filter_debug.append(f"{col} filter: Skipped (invalid min/max)")
                        logger.info(f"{col} filter skipped: Invalid min/max")
                else:
                    filter_debug.append(f"{col} filter: Skipped (all null)")
                    logger.info(f"{col} filter skipped: All null")

    # Categorical Filters
    with st.sidebar.expander("Categorical Filters", expanded=True):
        categorical_cols = ['Usage unit', 'Tax Percentage Valid', 'Quantity Valid', 'Final Amount Valid', 'Final Amount Matches', 'Tax Type', 'Customer Name']
        for col in categorical_cols:
            if col in filtered_df.columns:
                options = get_filter_options(filtered_df, col)
                if options:
                    select_all = st.checkbox(f"Select All {col}", value=True, key=f"select_all_{col}")
                    selected = st.multiselect(f"{col}", options, default=options if select_all else [],
                                             help=f"Filter by {col} values.", key=f"multiselect_{col}")
                    if selected:
                        prev_len = len(filtered_df)
                        filtered_df = filtered_df[filtered_df[col].isin([str(x) for x in selected]) | filtered_df[col].isna()]
                        filter_debug.append(f"{col} filter: {prev_len} -> {len(filtered_df)} rows")
                        active_filters.append(f"{col}: {len(selected)} selected")
                        logger.info(f"{col} filter applied: {len(filtered_df)} rows remain")
                    else:
                        filter_debug.append(f"{col} filter: Skipped (no selection)")
                        logger.info(f"{col} filter skipped: No selection")
                else:
                    filter_debug.append(f"{col} filter: Skipped (no valid values)")
                    logger.info(f"{col} filter skipped: No valid values")

    # Filter summary table
    with st.sidebar.expander("Active Filters Summary", expanded=True):
        if active_filters:
            filter_summary = pd.DataFrame(active_filters, columns=['Filter'])
            st.dataframe(filter_summary, use_container_width=True)
        else:
            st.write("No active filters applied.")

    # Update filtered rows count
    st.sidebar.write(f"Filtered Rows: {len(filtered_df)} (after filters)")

    # Check if filtered_df is empty
    if filtered_df.empty:
        st.warning("No data matches the current filters. Please adjust your filters.")
        st.write("Filter Debug Info:")
        for debug_msg in filter_debug:
            st.write(debug_msg)
        st.write(f"Sample Invoice No values (first 5): {df['Invoice No'].head().tolist()}")
        logger.warning("Filtered DataFrame is empty. Filter debug: " + "; ".join(filter_debug))
        st.stop()

    # Add total rows per invoice
    @st.cache_data
    def add_total_rows(df):
        try:
            total_rows = []
            for invoice in df['Invoice No'].unique():
                invoice_df = df[df['Invoice No'] == invoice]
                total_amount = invoice_df['Final Amount'].sum()
                total_row = pd.Series(index=df.columns, dtype=object)
                total_row['Invoice No'] = invoice
                total_row['Final Amount'] = total_amount
                total_row['Row Type'] = 'Total'
                total_rows.append(total_row)
                for _, row in invoice_df.iterrows():
                    total_rows.append(row)
            result_df = pd.DataFrame(total_rows).reset_index(drop=True)
            return result_df
        except Exception as e:
            st.error(f"Error adding total rows: {e}")
            logger.error(f"Total rows error: {e}")
            return df

    display_df = add_total_rows(filtered_df)

    # AI-Powered Anomaly Detection
    with st.expander("AI Anomaly Detection", expanded=False):
        st.subheader("Anomaly Detection")
        anomaly_cols = ['Tax Percentage Error', 'Final Amount Error', 'Quantity Error']
        if all(col in filtered_df.columns for col in anomaly_cols):
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(filtered_df[anomaly_cols].fillna(0))
            filtered_df['Is_Anomaly'] = anomalies == -1
            anomaly_df = filtered_df[filtered_df['Is_Anomaly']][['Invoice No', 'Item Name', 'Tax Percentage Error', 'Final Amount Error', 'Quantity Error']]
            st.write(f"Found {len(anomaly_df)} anomalies in error columns")
            if not anomaly_df.empty:
                st.dataframe(anomaly_df, use_container_width=True)
                fig = px.scatter(filtered_df, x='Quantity Error', y='Final Amount Error', color='Is_Anomaly',
                                hover_data=['Invoice No', 'Item Name'], title='Anomaly Detection in Errors')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Anomaly detection requires all error columns to be present.")

    # Display filtered data with dynamic columns
    with st.expander("Filtered Data", expanded=True):
        st.subheader("Filtered Data")
        available_cols = list(display_df.columns)
        default_cols = ['Invoice No', 'Bill Date', 'Quantity', 'Rate', 'Discount', 'Tax Percentage', 'Final Amount', 'Tax Percentage Valid', 'Quantity Valid', 'Final Amount Valid']
        selected_cols = st.multiselect("Select Columns to Display", available_cols, default=[col for col in default_cols if col in available_cols], key="display_cols")
        
        def highlight_errors(row):
            if row.get('Row Type') == 'Total':
                return ['background-color: lightgrey; font-weight: bold'] * len(row)
            color = 'yellow' if row.get('Tax Percentage Valid') == 'Invalid' else ''
            if row.get('Quantity Valid') == 'Invalid':
                color = 'orange'
            if row.get('Final Amount Valid') == 'Invalid':
                color = 'red'
            return [f'background-color: {color}'] * len(row)
        
        if selected_cols:
            st.dataframe(display_df[selected_cols].style.apply(highlight_errors, axis=1), use_container_width=True)
        else:
            st.warning("Please select at least one column to display.")

    # Download options
    with st.expander("Download Data", expanded=False):
        st.subheader("Download Data")
        try:
            csv = display_df.to_csv(index=False)
            st.download_button("Download as CSV", csv, "filtered_data_with_totals.csv", "text/csv")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                display_df.to_excel(writer, index=False, sheet_name='Filtered Data')
            excel_data = output.getvalue()
            st.download_button("Download as Excel", excel_data, "filtered_data_with_totals.xlsx", 
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Error generating download: {e}")
            logger.error(f"Download error: {e}")

    # Display formulas
    with st.expander("Calculation Formulas", expanded=False):
        st.subheader("Calculation Formulas")
        st.markdown("""
        - **Base Amount** = Quantity × Rate × (1 − Discount/100)
        - **Amount** = Quantity × Rate × (1 − Discount/100)
        - **Tax Amount** = Amount × Tax Percentage / 100
        - **Final Amount** = Amount + Tax Amount
        - **Calculated Tax Percentage** = (Tax Amount / Base Amount) × 100
        - **Tax Percentage Error** = Calculated Tax Percentage - Tax Percentage
        - **Tax Percentage Valid**: Valid if Tax Percentage in [0, 5, 12, 18, 28] and |Tax Percentage Error| < 0.1
        - **Calculated Quantity** = Amount / (Rate × (1 − Discount/100))
        - **Quantity Error** = |Calculated Quantity - Quantity|
        - **Quantity Valid**: Valid if Quantity Error < 0.01
        - **Calculated Final Amount** = Amount + Tax Amount
        - **Final Amount Error** = Calculated Final Amount - Final Amount
        - **Final Amount Valid**: Valid if |Final Amount Error| < 0.01
        - **Final Amount Matches**: Yes if Final Amount Valid but Tax Percentage Invalid, else No or N/A
        """)

    # Error summary
    with st.expander("Error Summary", expanded=False):
        st.subheader("Error Summary")
        error_summary = filtered_df.groupby('Invoice No').agg({
            'Tax Percentage Error': ['mean', 'sum'],
            'Quantity Error': ['mean', 'sum'],
            'Final Amount Error': ['mean', 'sum'],
            'Tax Percentage Valid': lambda x: (x == 'Invalid').sum(),
            'Quantity Valid': lambda x: (x == 'Invalid').sum(),
            'Final Amount Valid': lambda x: (x == 'Invalid').sum(),
            'Final Amount Matches': lambda x: (x == 'Yes').sum()
        }).reset_index()
        error_summary.columns = ['Invoice No', 'Avg Tax Percentage Error', 'Total Tax Percentage Error',
                                'Avg Quantity Error', 'Total Quantity Error',
                                'Avg Final Amount Error', 'Total Final Amount Error',
                                'Invalid Tax Percentage Count', 'Invalid Quantity Count',
                                'Invalid Final Amount Count', 'Final Amount Matches Count']
        st.dataframe(error_summary, use_container_width=True)

    # Summary statistics
    with st.expander("Summary Statistics", expanded=False):
        st.subheader("Summary Statistics")
        summary = {
            'Total Rows': len(filtered_df),
            'Rows with Invalid Tax Percentage': len(filtered_df[filtered_df['Tax Percentage Valid'] == 'Invalid']),
            'Rows with Invalid Quantity': len(filtered_df[filtered_df['Quantity Valid'] == 'Invalid']),
            'Rows with Invalid Final Amount': len(filtered_df[filtered_df['Final Amount Valid'] == 'Invalid']),
            'Rows where Final Amount Matches Despite Invalid Tax': len(filtered_df[filtered_df['Final Amount Matches'] == 'Yes']),
            'Total Quantity': filtered_df['Quantity'].sum(),
            'Total Final Amount': filtered_df['Final Amount'].sum(),
            'Total Tax Amount': filtered_df['Tax Amount'].sum(),
            'Average Discount (%)': filtered_df['Discount'].mean(),
            'Unique Customers': filtered_df['Customer Name'].nunique() if 'Customer Name' in filtered_df.columns else 0,
            'Unique Items': filtered_df['Item Name'].nunique() if 'Item Name' in filtered_df.columns else 0
        }
        st.write(summary)

    # Visualizations
    with st.expander("Visualizations", expanded=True):
        st.subheader("Visualizations")
        viz_option = st.selectbox("Select Visualization", [
            "Tax Percentage Error by Invoice", "Quantity Error by Invoice", "Final Amount Error by Invoice",
            "Final Amount Box Plot", "Quantity vs Final Amount", "Usage Unit Distribution",
            "Item Type Distribution", "Tax Type Distribution", "Purchase Trends Over Time",
            "Item Type Treemap", "Tax Flow Sankey"
        ], key="viz_select")

        try:
            viz_df = filtered_df
            
            if viz_option == "Tax Percentage Error by Invoice" and 'Tax Percentage Error' in viz_df.columns:
                fig = px.bar(viz_df, x='Invoice No', y='Tax Percentage Error', title='Tax Percentage Error by Invoice',
                            color='Tax Percentage Valid')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Quantity Error by Invoice" and 'Quantity Error' in viz_df.columns:
                fig = px.bar(viz_df, x='Invoice No', y='Quantity Error', title='Quantity Error by Invoice',
                            color='Quantity Valid')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Final Amount Error by Invoice" and 'Final Amount Error' in viz_df.columns:
                fig = px.bar(viz_df, x='Invoice No', y='Final Amount Error', title='Final Amount Error by Invoice',
                            color='Final Amount Valid')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Final Amount Box Plot" and 'Final Amount' in viz_df.columns:
                fig = px.box(viz_df, y='Final Amount', color='Item Type' if 'Item Type' in viz_df.columns else None,
                            title='Final Amount Box Plot', points='all')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Quantity vs Final Amount" and all(col in viz_df.columns for col in ['Quantity', 'Final Amount']):
                fig = px.scatter(viz_df, x='Quantity', y='Final Amount', color='Item Type' if 'Item Type' in viz_df.columns else None,
                                size='Tax Percentage', hover_data=['Item Name', 'Calculation Details'], 
                                title='Quantity vs Final Amount')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Usage Unit Distribution" and 'Usage unit' in viz_df.columns:
                fig = px.pie(viz_df, names='Usage unit', title='Distribution of Usage Unit')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Item Type Distribution" and 'Item Type' in viz_df.columns:
                fig = px.pie(viz_df, names='Item Type', title='Distribution of Item Type')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Tax Type Distribution" and 'Tax Type' in viz_df.columns:
                fig = px.histogram(viz_df, x='Tax Type', title='Tax Type Distribution', color='Tax Percentage Valid')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Purchase Trends Over Time" and 'Bill Date' in viz_df.columns:
                trends = viz_df.groupby(viz_df['Bill Date'].dt.to_period('M')).agg({
                    'Final Amount': ['sum', 'mean'],
                    'Tax Amount': 'sum'
                }).reset_index()
                trends['Bill Date'] = trends['Bill Date'].astype(str)
                trends.columns = ['Bill Date', 'Total Final Amount', 'Avg Final Amount', 'Total Tax Amount']
                fig = px.line(trends, x='Bill Date', y=['Total Final Amount', 'Avg Final Amount', 'Total Tax Amount'],
                             title='Purchase Trends Over Time')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Item Type Treemap" and 'Item Type' in viz_df.columns:
                fig = px.treemap(viz_df, path=['Item Type', 'Item Name'], values='Final Amount',
                                title='Item Type Treemap by Final Amount')
                st.plotly_chart(fig, use_container_width=True)

            elif viz_option == "Tax Flow Sankey" and all(col in viz_df.columns for col in ['Tax Type', 'Item Type']):
                sankey_data = viz_df.groupby(['Tax Type', 'Item Type'])['Final Amount'].sum().reset_index()
                labels = list(set(sankey_data['Tax Type'].unique()).union(set(sankey_data['Item Type'].unique())))
                source = [labels.index(t) for t in sankey_data['Tax Type']]
                target = [labels.index(t) for t in sankey_data['Item Type']]
                value = sankey_data['Final Amount']
                fig = go.Figure(data=[go.Sankey(
                    node=dict(label=labels),
                    link=dict(source=source, target=target, value=value)
                )])
                fig.update_layout(title_text="Tax Flow Sankey Diagram")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Visualization error: {e}")
            logger.error(f"Visualization error: {e}")

    # Advanced Analytics
    with st.expander("Advanced Analytics", expanded=True):
        st.subheader("Error Analysis Summary")
        error_cols = ['Tax Percentage Error', 'Final Amount Error', 'Quantity Error']
        selected_error_cols = st.multiselect("Select Error Columns to Analyze", error_cols, default=error_cols, key="error_cols")
        
        if selected_error_cols:
            @st.cache_data
            def compute_error_summary(df, cols):
                summary_data = []
                for col in cols:
                    data = {
                        'Column': col,
                        'Mean': df[col].mean(),
                        'Median': df[col].median(),
                        'Min': df[col].min(),
                        'Max': df[col].max(),
                        'Std Dev': df[col].std(),
                        'Non-Zero Errors': len(df[df[col] != 0])
                    }
                    summary_data.append(data)
                return pd.DataFrame(summary_data)
            
            summary_df = compute_error_summary(filtered_df, selected_error_cols)
            st.write("Error Summary Statistics")
            st.dataframe(summary_df, use_container_width=True)

        if 'Final Amount' in filtered_df.columns and not filtered_df.empty:
            st.subheader("Outlier Detection")
            q1 = filtered_df['Final Amount'].quantile(0.25)
            q3 = filtered_df['Final Amount'].quantile(0.75)
            iqr = q3 - q1
            outliers = filtered_df[(filtered_df['Final Amount'] < q1 - 1.5 * iqr) | (filtered_df['Final Amount'] > q3 + 1.5 * iqr)]
            st.write(f"Found {len(outliers)} outliers in Final Amount")
            if len(outliers) > 0:
                st.dataframe(outliers[['Invoice No', 'Item Name', 'Final Amount', 'Calculation Details']], use_container_width=True)

        if 'Customer Name' in filtered_df.columns and 'Final Amount' in filtered_df.columns and not filtered_df.empty:
            st.subheader("Customer Purchase Patterns")
            customer_summary = filtered_df.groupby('Customer Name').agg({
                'Final Amount': ['sum', 'mean', 'count'],
                'Item Name': 'nunique'
            }).reset_index()
            customer_summary.columns = ['Customer Name', 'Total Spent', 'Average Purchase', 'Purchase Count', 'Unique Items']
            st.dataframe(customer_summary, use_container_width=True)

        if all(col in filtered_df.columns for col in ['Quantity', 'Rate', 'Final Amount']) and len(filtered_df) >= 3:
            st.subheader("Item Clustering")
            features = filtered_df[['Quantity', 'Rate', 'Final Amount']].fillna(0)
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=min(3, len(features)), random_state=42)
            filtered_df['Cluster'] = kmeans.fit_predict(scaled_features)
            fig = px.scatter(filtered_df, x='Quantity', y='Final Amount', color='Cluster',
                            hover_data=['Item Name', 'Calculation Details'], 
                            title='Item Clustering by Quantity and Final Amount')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a .xlsx or .csv file to begin.")
