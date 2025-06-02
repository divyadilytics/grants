import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px
import time

# --- Configuration Settings for Snowflake and Cortex ---
HOST = "GBJYVCT-LSB50763.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.Grants_search_services"
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/grantsyaml_27.yaml'
MODELS = ["mistral-large", "snowflake-arctic", "llama3-70b", "llama3-8b"]

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Welcome to Cortex AI Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Utility Functions ---
def init_service_metadata():
    """Initialize metadata for Cortex Search services."""
    if not st.session_state.service_metadata:
        try:
            session = st.session_state.snowpark_session
            services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
            service_metadata = []
            if services:
                for s in services:
                    svc_name = s["name"]
                    svc_search_col = session.sql(
                        f"DESC CORTEX SEARCH SERVICE {svc_name};"
                    ).collect()[0]["search_column"]
                    service_metadata.append(
                        {"name": svc_name, "search_column": svc_search_col}
                    )
            st.session_state.service_metadata = service_metadata
        except Exception as e:
            st.error(f"❌ Failed to initialize Cortex Search service metadata: {str(e)}")
            st.session_state.service_metadata = [{"name": CORTEX_SEARCH_SERVICES, "search_column": ""}]

# Inject custom CSS to make the title static
st.markdown("""
    <style>
    .static-title {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: white;
        z-index: 1000;
        padding: 10px 20px;
        margin: 0;
        font-size: 24px;
        font-weight: bold;
        border-bottom: 1px solid #f0f2f6;
    }
    .content {
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Main UI ---
st.markdown('<div class="static-title">Cortex AI Assistant by DiLytics</div>', unsafe_allow_html=True)
semantic_model_filename = SEMANTIC_MODEL.split("/")[-1]
st.markdown(f"Semantic Model: `{semantic_model_filename}`")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.debug_mode = False
    st.session_state.last_suggestions = []
    st.session_state.chart_x_axis = None
    st.session_state.chart_y_axis = None
    st.session_state.chart_type = "Bar Chart"
    st.session_state.current_query = None
    st.session_state.current_results = None
    st.session_state.current_sql = None
    st.session_state.current_summary = None
    st.session_state.service_metadata = []
    st.session_state.selected_cortex_search_service = CORTEX_SEARCH_SERVICES
    st.session_state.model_name = "mistral-large"
    st.session_state.num_retrieved_chunks = 100
    st.session_state.num_chat_messages = 10
    st.session_state.use_chat_history = True
    st.session_state.clear_conversation = False
    st.session_state.show_selector = False
    st.session_state.show_greeting = True
    st.session_state.data_source = "Database"
    st.session_state.show_about = False
    st.session_state.show_help = False
    st.session_state.show_history = False
    st.session_state.query = None

# Wrap content in a div to apply margin
st.markdown('<div class="content">', unsafe_allow_html=True)

# Initialize service metadata after session state
if st.session_state.authenticated:
    init_service_metadata()

# --- Login Page ---
if not st.session_state.authenticated:
    st.title("Welcome to Snowflake Cortex AI")
    st.markdown("Please login to interact with your data")
    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user=st.session_state.username,
                password=st.session_state.password,
                account="GBJYVCT-LSB50763",
                host=HOST,
                port=443,
                warehouse="COMPUTE_WH",
                role="ACCOUNTADMIN",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn
            snowpark_session = Session.builder.configs({"connection": conn}).create()
            st.session_state.snowpark_session = snowpark_session
            with conn.cursor() as cur:
                cur.execute(f"USE DATABASE {DATABASE}")
                cur.execute(f"USE SCHEMA {SCHEMA}")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")
            st.session_state.authenticated = True
            st.success("Authentication successful! Redirecting...")
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
else:
    # Main app logic (abbreviated for brevity)
    session = st.session_state.snowpark_session
    root = Root(session)

    def run_snowflake_query(query):
        try:
            if not query:
                return None
            df = session.sql(query)
            data = df.collect()
            if not data:
                if st.session_state.debug_mode:
                    st.sidebar.warning("Query returned no data.")
                return None
            columns = df.schema.names
            result_df = pd.DataFrame(data, columns=columns)
            for col in result_df.columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    if "award" in col.lower() and "number" in col.lower():
                        result_df[col] = result_df[col].astype(float).astype(int).astype(str)
                    else:
                        result_df[col] = result_df[col].astype(float)
            if st.session_state.debug_mode:
                st.sidebar.text_area("Query Results", result_df.to_string(), height=200)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            return None

    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart", query: str = ""):
        try:
            if df is None or df.empty:
                st.warning("No valid data available for visualization.")
                return
            query_lower = query.lower()
            if re.search(r'\b(county|jurisdiction)\b', query_lower):
                default_data = "Pie Chart"
            elif re.search(r'\b(month|year|date)\b', query_lower):
                default_data = "Line Chart"
            else:
                default_data = "Bar Chart"
            all_cols = list(df.columns)
            col1, col2, col3 = st.columns(3)
            x_col = col1.selectbox("X axis", all_cols, index=0, key=f"{prefix}_x")
            remaining_cols = [c for c in all_cols if c != x_col]
            y_options = ["All Columns"] + remaining_cols
            y_col = col2.selectbox("Y axis", y_options, index=0, key=f"{prefix}_y")
            chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
            chart_type = col3.selectbox("Chart Type", chart_options, index=chart_options.index(default_data), key=f"{prefix}_type")
            axis_layout = {"xaxis": {"tickformat": "d"}, "yaxis": {"tickformat": "d"}}
            st.header(f"{chart_type} (All Columns)" if y_col == "All Columns" else chart_type)
            if y_col == "All Columns" and chart_type in ["Line Chart", "Bar Chart", "Scatter Chart"]:
                y_cols = remaining_cols
                if not y_cols:
                    st.warning("No Y-axis columns available for visualization.")
                    return
                df_melted = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="Category", value_name="Value")
                if chart_type == "Line Chart":
                    fig = px.line(df_melted, x=x_col, y="Value", color="Category")
                elif chart_type == "Bar Chart":
                    fig = px.bar(df_melted, x=x_col, y="Value", color="Category", barmode="group")
                elif chart_type == "Scatter Chart":
                    fig = px.scatter(df_melted, x=x_col, y="Value", color="Category")
                fig.update_layout(**axis_layout, title=None)
                st.plotly_chart(fig, key=f"{prefix}_{chart_type.lower().replace(' ', '_')}")
            else:
                if y_col == "All Columns":
                    y_col = remaining_cols[0] if remaining_cols else None
                    if not y_col:
                        st.warning("No Y-axis columns available for visualization.")
                        return
                if chart_type == "Pie Chart":
                    if len(df) > 0:
                        first_row = df.iloc[0]
                        pie_data = pd.DataFrame({
                            "Category": all_cols,
                            "Value": [first_row[col] for col in all_cols]
                        })
                        pie_data["Value"] = pd.to_numeric(pie_data["Value"], errors="coerce")
                        pie_data = pie_data.dropna(subset=["Value"])
                        pie_data = pie_data[pie_data["Value"] > 0]
                        if not pie_data.empty:
                            fig = px.pie(pie_data, names="Category", values="Value")
                            fig.update_layout(**axis_layout, title=None)
                            st.plotly_chart(fig, key=f"{prefix}_pie")
                        else:
                            st.warning("No positive numeric values available for Pie Chart.")
                    else:
                        st.warning("No data available for Pie Chart.")
                else:
                    if chart_type == "Line Chart":
                        fig = px.line(df, x=x_col, y=y_col)
                    elif chart_type == "Bar Chart":
                        fig = px.bar(df, x=x_col, y=y_col)
                    elif chart_type == "Scatter Chart":
                        fig = px.scatter(df, x=x_col, y=y_col)
                    elif chart_type == "Histogram Chart":
                        fig = px.histogram(df, x=x_col)
                    fig.update_layout(**axis_layout, title=None)
                    st.plotly_chart(fig, key=f"{prefix}_{chart_type.lower().replace(' ', '_')}")
        except Exception as e:
            st.error(f"❌ Error generating chart: {str(e)}")

    # Add more functions and logic here as needed (e.g., query processing, sidebar, etc.)
    # For brevity, only the critical parts are included above

st.markdown('</div>', unsafe_allow_html=True)  # Close content div
