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

# --- Initialize Session State ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
    st.session_state.chat_history = []
    st.session_state.messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = []
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None
if "service_metadata" not in st.session_state:
    st.session_state.service_metadata = []
if "selected_cortex_search_service" not in st.session_state:
    st.session_state.selected_cortex_search_service = CORTEX_SEARCH_SERVICES
if "model_name" not in st.session_state:
    st.session_state.model_name = "mistral-large"
if "num_retrieved_chunks" not in st.session_state:
    st.session_state.num_retrieved_chunks = 100
if "num_chat_messages" not in st.session_state:
    st.session_state.num_chat_messages = 10
if "use_chat_history" not in st.session_state:
    st.session_state.use_chat_history = True
if "clear_conversation" not in st.session_state:
    st.session_state.clear_conversation = False
if "show_selector" not in st.session_state:
    st.session_state.show_selector = False
if "show_greeting" not in st.session_state:
    st.session_state.show_greeting = True
if "data_source" not in st.session_state:
    st.session_state.data_source = "Database"
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "show_help" not in st.session_state:
    st.session_state.show_help = False
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "query" not in st.session_state:
    st.session_state.query = None

# --- Custom CSS ---
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
def stream_text(text: str, chunk_size: int = 2, delay: float = 0.04):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(delay)

def start_new_conversation():
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.current_query = None
    st.session_state.current_results = None
    st.session_state.current_sql = None
    st.session_state.current_summary = None
    st.session_state.chart_x_axis = None
    st.session_state.chart_y_axis = None
    st.session_state.chart_type = "Bar Chart"
    st.session_state.last_suggestions = []
    st.session_state.clear_conversation = False
    st.session_state.show_greeting = True
    st.session_state.query = None
    st.session_state.show_history = False
    st.rerun()

def init_service_metadata():
    if not st.session_state.service_metadata and st.session_state.snowpark_session:
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

def query_cortex_search_service(query):
    try:
        db, schema = session.get_current_database(), session.get_current_schema()
        root = Root(session)
        cortex_search_service = (
            root.databases[db]
            .schemas[schema]
            .cortex_search_services[st.session_state.selected_cortex_search_service]
        )
        context_documents = cortex_search_service.search(
            query, columns=[], limit=st.session_state.num_retrieved_chunks
        )
        results = context_documents.results
        service_metadata = st.session_state.service_metadata
        search_col = [s["search_column"] for s in service_metadata
                      if s["name"] == st.session_state.selected_cortex_search_service][0]
        context_str = ""
        for i, r in enumerate(results):
            context_str += f"Context document {i+1}: {r[search_col]} \n" + "\n"
        if st.session_state.debug_mode:
            st.sidebar.text_area("Context documents", context_str, height=500)
        return context_str
    except Exception as e:
        st.error(f"❌ Error querying Cortex Search service: {str(e)}")
        return ""

def get_chat_history():
    start_index = max(
        0, len(st.session_state.chat_history) - st.session_state.num_chat_messages
    )
    return st.session_state.chat_history[start_index : len(st.session_state.chat_history) - 1]

def make_chat_history_summary(chat_history, question):
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    prompt = f"""
        [INST]
        Based on the chat history below and the question, generate a query that extends the question
        with the chat history provided. The query should be in natural language.
        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history_str}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """
    summary = complete(st.session_state.model_name, prompt)
    if st.session_state.debug_mode:
        st.sidebar.text_area("Chat history summary", summary.replace("$", r"\$"), height=150)
    return summary

def create_prompt(user_question):
    chat_history_str = ""
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context = query_cortex_search_service(question_summary)
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        else:
            prompt_context = query_cortex_search_service(user_question)
    else:
        prompt_context = query_cortex_search_service(user_question)
        chat_history = []
    
    if not prompt_context.strip():
        return complete(st.session_state.model_name, user_question)
    
    prompt = f"""
        [INST]
        You are a helpful AI chat assistant with RAG capabilities. When a user asks you a question,
        you will also be given context provided between <context> and </context> tags. Use that context
        with the user's chat history provided in the between <chat_history> and </chat_history> tags
        to provide a summary that addresses the user's question. Ensure the answer is coherent, concise,
        and directly relevant to the user's question.

        If the user asks a generic question which cannot be answered with the given context or chat_history,
        just respond directly and concisely to the user's question using the LLM.

        <chat_history>
        {chat_history_str}
        </chat_history>
        <context>
        {prompt_context}
        </context>
        <question>
        {user_question}
        </question>
        [/INST]
        Answer:
    """
    return complete(st.session_state.model_name, prompt)

def get_user_questions(limit=10):
    user_questions = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    return user_questions[-limit:][::-1]

# --- Main Application Logic ---
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
            init_service_metadata()  # Call after authentication
            st.success("Authentication successful! Redirecting...")
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
else:
    session = st.session_state.snowpark_session
    root = Root(session)

    def run_snowflake_query(query):
        """Execute a Snowflake SQL query and return results as a DataFrame."""
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
            # Convert numerical columns appropriately
            for col in result_df.columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # Check if the column is an "award number" (adjust condition as needed)
                    if "award" in col.lower() and ("number" in col.lower() or "num" in col.lower() or "id" in col.lower()):
                        # Convert to integer and then to string to preserve exact value
                        result_df[col] = result_df[col].astype(float).astype(int).astype(str)
                    else:
                        # For other numeric columns, keep as float
                        result_df[col] = result_df[col].astype(float)
            if st.session_state.debug_mode:
                st.sidebar.text_area("Query Results", result_df.to_string(), height=200)
                st.sidebar.write("DataFrame dtypes:", result_df.dtypes)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            if st.session_state.debug_mode:
                st.sidebar.error(f"SQL Error Details: {str(e)}")
            return None

    def is_structured_query(query: str):
        structured_patterns = [
            r'\b(count|number|where|group\s+by|order\s+by|sum|avg|max|min|total|how\s+many|which|show|list|summary)\b',
            r'\b(award|budget|posted|encumbrance|date|task|actual|approved|total)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

    def is_complete_query(query: str):
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str):
        summarize_patterns = [r'\b(summarize|summary|condense)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str):
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|type of questions|queries)\b.*\b(ask|can i ask|pose)\b',
            r'\b(give me|show me|list)\b.*\b(questions|examples|sample questions)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def is_greeting_query(query: str):
        greeting_patterns = [
            r'^\b(hello|hi|hey|greet)\b$',
            r'^\b(hello|hi|hey,greet)\b\s.*$'
        ]
        return any(re.search(pattern, query.lower()) for pattern in greeting_patterns)

    def complete(model, prompt):
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            st.error(f"❌ COMPLETE Function Error: {str(e)}")
            return None

    def summarize(text):
        try:
            text = text.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query).collect()
            return result[0]["SUMMARY"]
        except Exception as e:
            st.error(f"❌ SUMMARIZE Function Error: {str(e)}")
            return None

    def parse_sse_response(response_text: str) -> List[Dict]:
        events = []
        lines = response_text.strip().split("\n")
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if data_str != "[DONE]":
                    try:
                        data_json = json.loads(data_str)
                        current_event["data"] = data_json
                        events.append(current_event)
                        current_event = {}
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Failed to parse SSE data: {str(e)} - Data: {data_str}")
        return events

    def process_sse_response(response, is_structured):
        sql = ""
        search_results = []
        if not response:
            return sql, search_results
        try:
            for event in response:
                if event.get("event") == "message.delta" and "data" in event:
                    delta = event["data"].get("delta", {})
                    content = delta.get("content", [])
                    for item in content:
                        if item.get("type") == "tool_results":
                            tool_results = item.get("tool_results", {})
                            if "content" in tool_results:
                                for result in tool_results["content"]:
                                    if result.get("type") == "json":
                                        result_data = result.get("json", {})
                                        if is_structured and "sql" in result_data:
                                            sql = result_data.get("sql", "")
                                        elif not is_structured and "searchResults" in result_data:
                                            search_results = [sr["text"] for sr in result_data["searchResults"]]
        except Exception as e:
            st.error(f"❌ Error Processing Response: {str(e)}")
        return sql.strip(), search_results

    def snowflake_api_call(query: str, is_structured: bool = False):
        payload = {
            "model": st.session_state.model_name,
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if is_structured:
            payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
            payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
        else:
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": st.session_state.selected_cortex_search_service, "max_results": st.session_state.num_retrieved_chunks}}
        try:
            resp = requests.post(
                url=f"https://{HOST}{API_ENDPOINT}",
                json=payload,
                headers={
                    "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
                    "Content-Type": "application/json"
                },
                timeout=API_TIMEOUT // 1000
            )
            if st.session_state.debug_mode:
                st.write(f"API Response Status: {resp.status_code}")
                st.write(f"API Raw Response: {resp.text}")
            if resp.status_code < 400:
                if not resp.text.strip():
                    st.error("❌ API returned an empty response.")
                    return None
                return parse_sse_response(resp.text)
            else:
                raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"❌ API Request Error: {str(e)}")
            return None

    def summarize_unstructured_answer(answer):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|")\s', answer)
        return "\n".join(f"• {sent.strip()}" for sent in sentences[:6])

    def suggest_sample_questions(query: str) -> List[str]:
        try:
            prompt = (
                f"The user asked: '{query}'. This question may be ambiguous or unclear in the context of a business-facing grants analytics assistant. "
                f"Generate 3–5 clear, concise sample questions related to grants, awards, budgets, or encumbrances. "
                f"The questions should be easy for a business user to understand and answerable using grants data such as award budgets, encumbrances, or dates. "
                f"Format as a numbered list. Example format:\n1. What is the total award budget posted by date?\n2. Which awards have the highest encumbrances?"
            )
            response = complete(st.session_state.model_name, prompt)
            if response:
                questions = []
                for line in response.split("\n"):
                    line = line.strip()
                    if re.match(r'^\d+\.\s*.+', line):
                        question = re.sub(r'^\d+\.\s*', '', line)
                        questions.append(question)
                return questions[:5]
            else:
                return [
                    "What is the total award budget posted by date?",
                    "Which awards have the highest encumbrances in the current quarter?",
                    "What is the total amount of award encumbrances approved this month?",
                    "What is the date-wise breakdown of award budgets?",
                    "Which awards have pending encumbrances for more than two weeks?",
                ]
        except Exception as e:
            st.error(f"❌ Failed to generate sample questions: {str(e)}")
            return [
                "What is the total award budget posted by date?",
                "Which awards have the highest encumbrances in the current quarter?",
                "What is the total amount of award encumbrances approved this month?",
                "What is the date-wise breakdown of award budgets?",
                "Which awards have pending encumbrances for more than two weeks?",
            ]

    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart", query: str = ""):
        """Display a chart based on query results with user-selected options."""
        try:
            if df is None or df.empty or len(df.columns) < 2:
                st.warning("No valid data available for visualization.")
                if st.session_state.debug_mode:
                    st.sidebar.warning(f"Chart Data Issue: df={df}, columns={df.columns if df is not None else 'None'}")
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
            # Add "All Columns" option to Y-axis
            y_options = ["All Columns"] + remaining_cols
            y_col = col2.selectbox("Y axis", y_options, index=0, key=f"{prefix}_y")
            chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
            chart_type = col3.selectbox("Chart Type", chart_options, index=chart_options.index(default_data), key=f"{prefix}_type")

            if st.session_state.debug_mode:
                st.sidebar.text_area("Chart Config", f"X: {x_col}, Y: {y_col}, Type: {chart_type}", height=100)
                st.sidebar.write("DataFrame dtypes:", df.dtypes)

            # Configure axis formatting to avoid 'k' suffix
            axis_layout = {
                "xaxis": {"tickformat": "d"},
                "yaxis": {"tickformat": "d"}
            }

            # Handle "All Columns" for Y-axis
            if y_col == "All Columns":
                numeric_cols = [c for c in remaining_cols if pd.api.types.is_numeric_dtype(df[c])]
                if not numeric_cols:
                    st.warning("No numeric columns available for 'All Columns' visualization.")
                    if st.session_state.debug_mode:
                        st.sidebar.warning("No numeric columns found for 'All Columns'.")
                    return
                if chart_type in ["Line Chart", "Bar Chart", "Scatter Chart"]:
                    y_cols = numeric_cols
                    # Melt the DataFrame to long format for plotting multiple Y columns
                    df_melted = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="Category", value_name="Value")
                    if chart_type == "Line Chart":
                        fig = px.line(df_melted, x=x_col, y="Value", color="Category", title=f"{chart_type} (All Columns)")
                        fig.update_layout(**axis_layout)
                        st.plotly_chart(fig, key=f"{prefix}_line")
                    elif chart_type == "Bar Chart":
                        fig = px.bar(df_melted, x=x_col, y="Value", color="Category", title=f"{chart_type} (All Columns)", barmode="group")
                        fig.update_layout(**axis_layout)
                        st.plotly_chart(fig, key=f"{prefix}_bar")
                    elif chart_type == "Scatter Chart":
                        fig = px.scatter(df_melted, x=x_col, y="Value", color="Category", title=f"{chart_type} (All Columns)")
                        fig.update_layout(**axis_layout)
                        st.plotly_chart(fig, key=f"{prefix}_scatter")
                elif chart_type == "Pie Chart":
                    # For Pie Chart, sum all numeric columns (except X-axis) for each row
                    if len(df) > 0:
                        df['Total'] = df[numeric_cols].sum(axis=1)
                        fig = px.pie(df, names=x_col, values='Total', title=f"{chart_type} (All Columns)")
                        fig.update_layout(**axis_layout)
                        st.plotly_chart(fig, key=f"{prefix}_pie")
                    else:
                        st.warning("No data available for Pie Chart visualization.")
                else:
                    st.warning(f"'All Columns' option is not supported for {chart_type}.")
            else:
                # Single Y-column selection
                if chart_type == "Line Chart":
                    fig = px.line(df, x=x_col, y=y_col, title=chart_type)
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_line")
                elif chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col, title=chart_type)
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_bar")
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=x_col, values=y_col, title=chart_type)
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_pie")
                elif chart_type == "Scatter Chart":
                    fig = px.scatter(df, x=x_col, y=y_col, title=chart_type)
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_scatter")
                elif chart_type == "Histogram Chart":
                    fig = px.histogram(df, x=x_col, title=chart_type)
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_hist")
        except Exception as e:
            st.error(f"❌ Error generating chart: {str(e)}")
            if st.session_state.debug_mode:
                st.sidebar.error(f"Chart Error Details: {str(e)}")

    def toggle_about():
        st.session_state.show_about = not st.session_state.show_about
        st.session_state.show_help = False
        st.session_state.show_history = False

    def toggle_help():
        st.session_state.show_help = not st.session_state.show_help
        st.session_state.show_about = False
        st.session_state.show_history = False

    def toggle_history():
        st.session_state.show_history = not st.session_state.show_history
        st.session_state.show_about = False
        st.session_state.show_help = False

    # --- Sidebar Content ---
    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] [data-testid="stButton"] > button {
            background-color: #29B5E8 !important;
            color: white !important;
            font-weight: bold !important;
            width: 100% !important;
            border-radius: 0px !important;
            margin: 0 !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"][aria-label="Clear conversation"] > button,
        [data-testid="stSidebar"] [data-testid="stButton"][aria-label="About"] > button,
        [data-testid="stSidebar"] [data-testid="stButton"][aria-label="Help & Documentation"] > button,
        [data-testid="stSidebar"] [data-testid="stButton"][aria-label="History"] > button {
            background-color: #28A745 !important;
            color: white !important;
            font-weight: normal !important;
            border: 1px solid #28A745 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        logo_url = "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg"
        st.image(logo_url, width=250)

        if st.button("Clear conversation", key="clear_conversation_button"):
            start_new_conversation()

        st.radio("Select Data Source:", ["Database", "Document"], key="data_source")

        st.selectbox(
            "Select cortex search service:",
            [s["name"] for s in st.session_state.service_metadata] or [CORTEX_SEARCH_SERVICES],
            key="selected_cortex_search_service"
        )
        st.toggle("Debug", key="debug_mode", value=st.session_state.debug_mode)
        st.toggle("Use chat history", key="use_chat_history", value=True)
        with st.expander("Advanced options"):
            st.selectbox("Select model:", MODELS, key="model_name")
            st.number_input(
                "Select number of context chunks",
                value=100,
                key="num_retrieved_chunks",
                min_value=1,
                max_value=400
            )
            st.number_input(
                "Select number of messages to use in chat history",
                value=10,
                key="num_chat_messages",
                min_value=1,
                max_value=100
            )
        if st.session_state.debug_mode:
            st.expander("Session State").write(st.session_state)

        if st.button("Sample Questions", key="sample_questions_button"):
            st.session_state.show_sample_questions = not st.session_state.get("show_sample_questions", False)
        if st.session_state.get("show_sample_questions", False):
            sample_questions = [
                "What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?",
                "Give me date wise award breakdowns",
                "Give me award breakdowns",
                "Give me date wise award budget, actual award posted, award encumbrance posted, award encumbrance approved",
                "What is the task actual posted by award name?",
                "What is the award budget posted by date for these awards?",
                "What is the total award encumbrance posted for these awards?",
                "What is the total amount of award encumbrances approved?",
                "What is the total actual award posted for these awards?",
                "what is the award budget posted?",
                "what is this document about",
                "Subject areas",
                "explain five layers in High level Architecture"
            ]
            for sample in sample_questions:
                if st.button(sample, key=f"sidebar_{sample}"):
                    st.session_state.query = sample
                    st.session_state.show_greeting = False

        st.markdown("---")

        if st.button("History", key="history_button"):
            toggle_history()
        if st.session_state.show_history:
            st.markdown("### Recent Questions")
            user_questions = get_user_questions(limit=10)
            if not user_questions:
                st.write("No questions in history yet.")
            else:
                for idx, question in enumerate(user_questions):
                    if st.button(question, key=f"history_{idx}"):
                        st.session_state.query = question
                        st.session_state.show_greeting = False

        if st.button("About", key="about_button"):
            toggle_about()
        if st.session_state.show_about:
            st.markdown("### About")
            st.write(
                "This application uses **Snowflake Cortex Analyst** to interpret "
                "your natural language questions and generate data insights. "
                "Simply ask a question below to see relevant answers and visualizations."
            )

        if st.button("Help & Documentation", key="help_button"):
            toggle_help()
        if st.session_state.show_help:
            st.markdown("### Help & Documentation")
            st.write(
                "- [User Guide](https://docs.snowflake.com/en/guides-overview-ai-features)  \n"
                "- [Snowflake Cortex Analyst Docs](https://docs.snowflake.com/)  \n"
                "- [Contact Support](https://www.snowflake.com/en/support/)"
            )

    # --- Main UI ---
    st.title("Cortex AI Assistant by DiLytics")
    semantic_model_filename = SEMANTIC_MODEL.split("/")[-1]
    st.markdown(f"Semantic Model: `{semantic_model_filename}`")
    init_service_metadata()

    if st.session_state.show_greeting and not st.session_state.chat_history:
        st.markdown("Welcome! I’m the Snowflake AI Assistant, ready to assist you with grant data analysis, summaries, and answers — simply type your question to get started.")
    else:
        st.session_state.show_greeting = False

    for message in st.session_state.chat_history:
        try:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                if message["role"] == "assistant" and "results" in message and message["results"] is not None:
                    with st.expander("View SQL Query", expanded=False):
                        st.code(message["sql"], language="sql")
                    # Display DataFrame with custom formatting for award numbers
                    st.markdown(f"**Query Results ({len(message['results'])}) rows):**")
                    column_config = {
                        col: st.column_config.NumberColumn(format="%d")  # Plain integer format
                        for col in message["results"].columns
                        if pd.api.types.is_numeric_dtype(message["results"][col])
                        and not ("award" in col.lower() and ("number" in col.lower() or "num" in col.lower() or "id" in col.lower()))
                    }
                    st.dataframe(message["results"], column_config=column_config)
                    if not message["results"].empty and len(message["results"].columns) >= 2:
                        st.markdown("**📈 Visualization:**")
                        display_chart_tab(message["results"], prefix=f"chart_{hash(message['content'])}", query=message.get("query", ""))
        except Exception as e:
            st.error(f"Error displaying message: {str(e)}")

    chat_input_query = st.chat_input("Ask your question...")
    if chat_input_query:
        st.session_state.query = chat_input_query

    if st.session_state.query:
        try:
            query = st.session_state.query
            if query.lower().startswith("no of"):
                query = query.replace("no of", "number of", 1)
            
            st.session_state.show_greeting = False
            st.session_state.chart_x_axis = None
            st.session_state.chart_y_axis = None
            st.session_state.chart_type = "Bar Chart"
            original_query = query
            if query.strip().isdigit() and st.session_state.last_suggestions:
                try:
                    index = int(query.strip()) - 1
                    if 0 <= index < len(st.session_state.last_suggestions):
                        query = st.session_state.last_suggestions[index]
                    else:
                        query = original_query
                except ValueError:
                    query = original_query
            st.session_state.chat_history.append({"role": "user", "content": original_query})
            st.session_state.messages.append({"role": "user", "content": original_query})
            with st.chat_message("user"):
                st.markdown(original_query)
            with st.chat_message("assistant"):
                with st.spinner("Generating Response..."):
                    try:
                        response_placeholder = st.empty()
                        if st.session_state.data_source not in ["Database", "Document"]:
                            st.session_state.data_source = "Database"
                        is_structured = is_structured_query(query) and st.session_state.data_source == "Database"
                        is_complete = is_complete_query(query)
                        is_summarize = is_summarize_query(query)
                        is_suggestion = is_question_suggestion_query(query)
                        is_greeting = is_greeting_query(query)
                        if st.session_state.debug_mode:
                            st.sidebar.text_area("Debug Info", f"is_structured: {is_structured}\nData Source: {st.session_state.data_source}", height=100)
                        assistant_response = {"role": "assistant", "content": "", "query": query}
                        response_content = ""
                        failed_response = False

                        if is_greeting or is_suggestion:
                            suggested_questions = [
                                "What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?",
                                "Give me date-wise award breakdowns.",
                                "What is this document about?",
                                "List all subject areas."
                            ]
                            response_content = (
                                "Hello! Welcome to the GRANTS AI Assistant!\n"
                                "I'm here to help you explore and analyze grant-related data, answer questions about awards, budgets, and more, or provide insights from documents.\n\n"
                                "Here are some questions you can try:\n\n" +
                                "\n".join([f"{i}. {question}" for i, question in enumerate(suggested_questions, 1)]) +
                                "\n\nFeel free to ask anything, or pick one of the suggested questions to get started!"
                            )
                            with response_placeholder:
                                st.markdown(response_content, unsafe_allow_html=True)
                            assistant_response["content"] = response_content
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response["content"]})
                            st.session_state.last_suggestions = suggested_questions

                        elif is_complete:
                            response = create_prompt(query)
                            if response:
                                response_content = f"**✍️ Generated Response:**\n{response}"
                                with response_placeholder:
                                    st.write_stream(stream_text(response_content))
                                    st.markdown(response_content, unsafe_allow_html=True)
                                assistant_response["content"] = response_content
                                st.session_state.messages.append({"role": "assistant", "content": response_content})
                            else:
                                response_content = ""
                                failed_response = True
                                assistant_response["content"] = response_content

                        elif is_summarize:
                            summary = summarize(query)
                            if summary:
                                response_content = f"**Summary:**\n{summary}"
                                with response_placeholder:
                                    st.write_stream(stream_text(response_content))
                                    st.markdown(response_content, unsafe_allow_html=True)
                                assistant_response["content"] = response_content
                                st.session_state.messages.append({"role": "assistant", "content": response_content})
                            else:
                                response_content = ""
                                failed_response = True
                                assistant_response["content"] = response_content

                        elif st.session_state.data_source == "Database" and is_structured:
                            response = snowflake_api_call(query, is_structured=True)
                            sql, _ = process_sse_response(response, is_structured=True)
                            if sql:
                                if st.session_state.debug_mode:
                                    st.sidebar.text_area("Generated SQL", sql, height=150)
                                results = run_snowflake_query(sql)
                                if results is not None and not results.empty:
                                    results_text = results.to_string(index=False)
                                    prompt = f"Provide a concise natural language answer to the query '{query}' using the following data, avoiding phrases like 'Based on the query results':\n\n{results_text}"
                                    summary = complete(st.session_state.model_name, prompt)
                                    if not summary:
                                        summary = "⚠️ Unable to generate a natural language summary."
                                    response_content = f"**✍️ Generated Response:**\n{summary}"
                                    with response_placeholder:
                                        st.write_stream(stream_text(response_content))
                                        st.markdown(response_content, unsafe_allow_html=True)
                                    with st.expander("View SQL Query", expanded=False):
                                        st.code(sql, language="sql")
                                    # Display DataFrame with custom formatting
                                    st.markdown(f"**Query Results ({len(results)} rows):**")
                                    column_config = {
                                        col: st.column_config.NumberColumn(format="%d")
                                        for col in results.columns
                                        if pd.api.types.is_numeric_dtype(results[col])
                                        and not ("award" in col.lower() and ("number" in col.lower() or "num" in col.lower() or "id" in col.lower()))
                                    }
                                    st.dataframe(results, column_config=column_config)
                                    if len(results.columns) >= 2:
                                        st.markdown("**📈 Visualization:**")
                                        display_chart_tab(results, prefix=f"chart_{hash(query)}", query=query)
                                    assistant_response.update({
                                        "content": response_content,
                                        "sql": sql,
                                        "results": results,
                                        "summary": summary
                                    })
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": response_content,
                                        "sql": sql,
                                        "results": results,
                                        "summary": summary
                                    })
                                else:
                                    response_content = "No data returned for the query."
                                    failed_response = True
                                    assistant_response["content"] = response_content
                            else:
                                response_content = "Failed to generate SQL query."
                                failed_response = True
                                assistant_response["content"] = response_content

                        elif st.session_state.data_source == "Document":
                            response = snowflake_api_call(query, is_structured=False)
                            _, search_results = process_sse_response(response, is_structured=False)
                            if search_results:
                                raw_result = search_results[0]
                                summary = create_prompt(query)
                                if summary:
                                    response_content = f"**Here is the Answer:**\n{summary}"
                                    with response_placeholder:
                                        st.write_stream(stream_text(response_content))
                                        st.markdown(response_content, unsafe_allow_html=True)
                                    assistant_response["content"] = response_content
                                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                                else:
                                    response_content = f"**🔍 Key Information (Unsummarized):**\n{summarize_unstructured_answer(raw_result)}"
                                    with response_placeholder:
                                        st.write_stream(stream_text(response_content))
                                        st.markdown(response_content, unsafe_allow_html=True)
                                    assistant_response["content"] = response_content
                                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                            else:
                                response_content = ""
                                failed_response = True
                                assistant_response["content"] = response_content

                        else:
                            response_content = "Please select a data source to proceed with your query."
                            with response_placeholder:
                                st.write_stream(stream_text(response_content))
                                st.markdown(response_content, unsafe_allow_html=True)
                            assistant_response["content"] = response_content
                            st.session_state.messages.append({"role": "assistant", "content": response_content})

                        if failed_response:
                            suggestions = suggest_sample_questions(query)
                            response_content = "I am not sure about your question. Here are some questions you can ask me:\n\n"
                            for i, suggestion in enumerate(suggestions, 1):
                                response_content += f"{i}. {suggestion}\n"
                            response_content += "\nThese questions might help clarify your query. Feel free to try one or rephrase your question!"
                            with response_placeholder:
                                st.write_stream(stream_text(response_content))
                                st.markdown(response_content, unsafe_allow_html=True)
                            assistant_response["content"] = response_content
                            st.session_state.last_suggestions = suggestions
                            st.session_state.messages.append({"role": "assistant", "content": response_content})

                        st.session_state.chat_history.append(assistant_response)
                        st.session_state.current_query = query
                        st.session_state.current_results = assistant_response.get("results")
                        st.session_state.current_sql = assistant_response.get("sql")
                        st.session_state.current_summary = assistant_response.get("summary")
                        st.session_state.query = None
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error handling query: {str(e)}")
