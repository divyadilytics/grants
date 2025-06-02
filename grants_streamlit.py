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
# Snowflake connection details
HOST = "GBJYVCT-LSB50763.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"  # Endpoint for Cortex API calls
API_TIMEOUT = 50000  # Timeout for API calls in milliseconds
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.Grants_search_services"  # Default Cortex search service
CECON_SEARCH_SERVICES = "AI.DWH_MART.Grants_search_services"  # Alternative search service (unused)
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/grantsyaml_27.yaml'  # Semantic model for SQL generation

# Available LLMs for Cortex
MODELS = [
    "mistral-large",
    "snowflake-arctic",
    "llama3-70b",
    "llama3-8b",
]

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Welcome to Cortex AI Assistant",
    layout="wide",  # Use wide layout for better UI space
    initial_sidebar_state="auto"
)

# --- Initialize Session State ---
# Ensure all session state variables are initialized to manage app state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False  # Track login status
    st.session_state.username = ""  # Store Snowflake username
    st.session_state.password = ""  # Store Snowflake password
    st.session_state.CONN = None  # Snowflake connection object
    st.session_state.snowpark_session = None  # Snowpark session for Snowflake
    st.session_state.chat_history = []  # Store chat history (user and assistant messages)
    st.session_state.messages = []  # Store messages for display
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False  # Enable/disable debug mode
if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = []  # Store last suggested questions
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None  # X-axis for charts
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None  # Y-axis for charts
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"  # Default chart type
if "current_query" not in st.session_state:
    st.session_state.current_query = None  # Store the current user query
if "current_results" not in st.session_state:
    st.session_state.current_results = None  # Store query results
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None  # Store generated SQL
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None  # Store summarized response
if "service_metadata" not in st.session_state:
    st.session_state.service_metadata = []  # Metadata for Cortex search services
if "selected_cortex_search_service" not in st.session_state:
    st.session_state.selected_cortex_search_service = CORTEX_SEARCH_SERVICES  # Default search service
if "model_name" not in st.session_state:
    st.session_state.model_name = "mistral-large"  # Default LLM model
if "num_retrieved_chunks" not in st.session_state:
    st.session_state.num_retrieved_chunks = 100  # Number of context chunks to retrieve
if "num_chat_messages" not in st.session_state:
    st.session_state.num_chat_messages = 10  # Number of chat messages to use in history
if "use_chat_history" not in st.session_state:
    st.session_state.use_chat_history = True  # Enable/disable chat history usage
if "clear_conversation" not in st.session_state:
    st.session_state.clear_conversation = False  # Track if conversation should be cleared
if "show_selector" not in st.session_state:
    st.session_state.show_selector = False  # Show/hide selector (unused)
if "show_greeting" not in st.session_state:
    st.session_state.show_greeting = True  # Show initial greeting
if "data_source" not in st.session_state:
    st.session_state.data_source = "Database"  # Default data source (Database/Document)
if "show_about" not in st.session_state:
    st.session_state.show_about = False  # Show/hide About section
if "show_help" not in st.session_state:
    st.session_state.show_help = False  # Show/hide Help section
if "show_history" not in st.session_state:
    st.session_state.show_history = False  # Show/hide History section
if "query" not in st.session_state:
    st.session_state.query = None  # Store the current query from user input

# --- Custom CSS ---
# Hide Streamlit branding and prevent chat history shading for a cleaner UI
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
    """Stream text in chunks for a typewriter effect in the UI."""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(delay)

def start_new_conversation():
    """Reset the conversation state to start a new session."""
    st.session_state.chat_history = []  # Clear chat history
    st.session_state.messages = []  # Clear displayed messages
    st.session_state.current_query = None  # Clear current query
    st.session_state.current_results = None  # Clear query results
    st.session_state.current_sql = None  # Clear generated SQL
    st.session_state.current_summary = None  # Clear summarized response
    st.session_state.chart_x_axis = None  # Reset chart X-axis
    st.session_state.chart_y_axis = None  # Reset chart Y-axis
    st.session_state.chart_type = "Bar Chart"  # Reset chart type
    st.session_state.last_suggestions = []  # Clear last suggestions
    st.session_state.clear_conversation = False  # Reset clear conversation flag
    st.session_state.show_greeting = True  # Show greeting message
    st.session_state.query = None  # Clear current query
    st.session_state.show_history = False  # Hide history
    st.rerun()  # Rerun the app to apply changes

def init_service_metadata():
    """Initialize metadata for Cortex Search services."""
    if not st.session_state.service_metadata:
        try:
            # Fetch available Cortex Search services
            services = session.sql("SHOW CORTEX SEARCH SERVICES;").collect()
            service_metadata = []
            if services:
                for s in services:
                    svc_name = s["name"]
                    # Get the search column for each service
                    svc_search_col = session.sql(
                        f"DESC CORTEX SEARCH SERVICE {svc_name};"
                    ).collect()[0]["search_column"]
                    service_metadata.append(
                        {"name": svc_name, "search_column": svc_search_col}
                    )
            st.session_state.service_metadata = service_metadata
        except Exception as e:
            # Fallback to default service if fetching metadata fails
            st.error(f"❌ Failed to initialize Cortex Search service metadata: {str(e)}")
            st.session_state.service_metadata = [{"name": CORTEX_SEARCH_SERVICES, "search_column": ""}]

def init_config_options():
    """Placeholder function for config options (moved to sidebar)."""
    pass

def query_cortex_search_service(query):
    """Query the Cortex Search service to retrieve relevant context documents."""
    try:
        db, schema = session.get_current_database(), session.get_current_schema()
        root = Root(session)
        # Access the selected Cortex Search service
        cortex_search_service = (
            root.databases[db]
            .schemas[schema]
            .cortex_search_services[st.session_state.selected_cortex_search_service]
        )
        # Perform the search with the given query
        context_documents = cortex_search_service.search(
            query, columns=[], limit=st.session_state.num_retrieved_chunks
        )
        results = context_documents.results
        service_metadata = st.session_state.service_metadata
        # Extract the search column for the selected service
        search_col = [s["search_column"] for s in service_metadata
                      if s["name"] == st.session_state.selected_cortex_search_service][0]
        context_str = ""
        # Format the search results as a string
        for i, r in enumerate(results):
            context_str += f"Context document {i+1}: {r[search_col]} \n" + "\n"
        if st.session_state.debug_mode:
            st.sidebar.text_area("Context documents", context_str, height=500)
        return context_str
    except Exception as e:
        st.error(f"❌ Error querying Cortex Search service: {str(e)}")
        return ""

def get_chat_history():
    """Retrieve the most recent chat history messages."""
    start_index = max(
        0, len(st.session_state.chat_history) - st.session_state.num_chat_messages
    )
    return st.session_state.chat_history[start_index : len(st.session_state.chat_history) - 1]

def make_chat_history_summary(chat_history, question):
    """Generate a summary of the chat history to extend the current question."""
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
    """Create a prompt for the LLM, incorporating chat history and context."""
    chat_history_str = ""
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            # Summarize chat history and retrieve context
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
    
    # Construct the prompt with chat history, context, and question
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
    """
    Extract the last 'limit' user questions from the chat history.
    Returns a list of questions in reverse chronological order (most recent first).
    """
    user_questions = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    return user_questions[-limit:][::-1]

# --- Main Application Logic ---
if not st.session_state.authenticated:
    # --- Login Page ---
    st.title("Welcome to Snowflake Cortex AI")
    st.markdown("Please login to interact with your data")
    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            # Connect to Snowflake
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
            # Create a Snowpark session
            snowpark_session = Session.builder.configs({
                "connection": conn
            }).create()
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
            # Convert numerical columns to plain integers to avoid 'k' formatting
            for col in result_df.columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # Check if the column name suggests it's an identifier like "award number"
                    if "award" in col.lower() and "number" in col.lower():
                        # Convert to integer and then to string to preserve exact value
                        result_df[col] = result_df[col].astype(float).astype(int).astype(str)
                    else:
                        # For other numeric columns, ensure they are floats without formatting
                        result_df[col] = result_df[col].astype(float)
            if st.session_state.debug_mode:
                st.sidebar.text_area("Query Results", result_df.to_string(), height=200)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            if st.session_state.debug_mode:
                st.sidebar.error(f"SQL Error Details: {str(e)}")
            return None

    def is_structured_query(query: str):
        """Check if the query is structured (likely to generate SQL)."""
        structured_patterns = [
            r'\b(count|number|where|group\s+by|order\s+by|sum|avg|max|min|total|how\s+many|which|show|list|summary)\b',
            r'\b(award|budget|posted|encumbrance|date|task|actual|approved|total)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

    def is_complete_query(query: str):
        """Check if the query requires content generation (e.g., write, explain)."""
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str):
        """Check if the query requests a summary."""
        summarize_patterns = [r'\b(summarize|summary|condense)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str):
        """Check if the query asks for suggested questions."""
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|type of questions|queries)\b.*\b(ask|can i ask|pose)\b',
            r'\b(give me|show me|list)\b.*\b(questions|examples|sample questions)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def is_greeting_query(query: str):
        """Check if the query is a greeting (e.g., hi, hello)."""
        greeting_patterns = [
            r'^\b(hello|hi|hey|greet)\b$',
            r'^\b(hello|hi|hey,greet)\b\s.*$'
        ]
        return any(re.search(pattern, query.lower()) for pattern in greeting_patterns)

    def complete(model, prompt):
        """Call Snowflake Cortex COMPLETE function to generate a response."""
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            st.error(f"❌ COMPLETE Function Error: {str(e)}")
            return None

    def summarize(text):
        """Call Snowflake Cortex SUMMARIZE function to summarize text."""
        try:
            text = text.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query).collect()
            return result[0]["SUMMARY"]
        except Exception as e:
            st.error(f"❌ SUMMARIZE Function Error: {str(e)}")
            return None

    def parse_sse_response(response_text: str) -> List[Dict]:
        """Parse Server-Sent Events (SSE) response from Cortex API."""
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
        """Process SSE response to extract SQL or search results."""
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
        """Make an API call to Snowflake Cortex for structured or unstructured queries."""
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
        """Summarize an unstructured answer into bullet points."""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|")\s', answer)
        return "\n".join(f"• {sent.strip()}" for sent in sentences[:6])

    def suggest_sample_questions(query: str) -> List[str]:
        """Generate sample questions when a query fails or is unclear."""
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
            if df is None or df.empty:
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
            # Add "All Columns" option to the Y-axis dropdown
            y_options = ["All Columns"] + remaining_cols
            y_col = col2.selectbox("Y axis", y_options, index=0, key=f"{prefix}_y")
            chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
            chart_type = col3.selectbox("Chart Type", chart_options, index=chart_options.index(default_data), key=f"{prefix}_type")

            if st.session_state.debug_mode:
                st.sidebar.text_area("Chart Config", f"X: {x_col}, Y: {y_col}, Type: {chart_type}", height=100)

            # Configure axis formatting to avoid 'k' suffix for numerical values
            axis_layout = {
                "xaxis": {"tickformat": "d"},  # Use plain numbers for X-axis
                "yaxis": {"tickformat": "d"}   # Use plain numbers for Y-axis
            }

            # Handle "All Columns" selection for Y-axis
            if y_col == "All Columns" and chart_type in ["Line Chart", "Bar Chart", "Scatter Chart"]:
                # Plot X-axis against all other columns
                y_cols = remaining_cols
                if not y_cols:
                    st.warning("No Y-axis columns available for visualization.")
                    return
                if chart_type == "Line Chart":
                    # Melt the DataFrame to long format for plotting multiple Y columns
                    df_melted = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="Category", value_name="Value")
                    fig = px.line(df_melted, x=x_col, y="Value", color="Category", title="Line Chart (All Columns)")
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_line")
                elif chart_type == "Bar Chart":
                    # Melt the DataFrame to long format for plotting multiple Y columns
                    df_melted = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="Category", value_name="Value")
                    fig = px.bar(df_melted, x=x_col, y="Value", color="Category", title="Bar Chart (All Columns)", barmode="group")
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_bar")
                elif chart_type == "Scatter Chart":
                    # Melt the DataFrame to long format for plotting multiple Y columns
                    df_melted = df.melt(id_vars=[x_col], value_vars=y_cols, var_name="Category", value_name="Value")
                    fig = px.scatter(df_melted, x=x_col, y="Value", color="Category", title="Scatter Chart (All Columns)")
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_scatter")
            else:
                # Handle single Y-column selection (including fallback for Pie Chart and Histogram Chart)
                if y_col == "All Columns":
                    # Default to the first remaining column for chart types that don't support multiple Y columns
                    y_col = remaining_cols[0] if remaining_cols else None
                    if not y_col:
                        st.warning("No Y-axis columns available for visualization.")
                        return

                if chart_type == "Pie Chart":
                    # For pie chart, use column names as labels and values from the first row
                    if len(df) > 0:
                        try:
                            # Take the first row of the dataframe
                            first_row = df.iloc[0]
                            # Create a new dataframe with column names as labels and their values
                            pie_data = pd.DataFrame({
                                "Category": all_cols,
                                "Value": [first_row[col] for col in all_cols]
                            })
                            # Convert Value column to numeric, coercing errors to NaN
                            pie_data["Value"] = pd.to_numeric(pie_data["Value"], errors="coerce")
                            # Drop rows with NaN values in Value
                            pie_data = pie_data.dropna(subset=["Value"])
                            # Filter out zero or negative values for better visualization
                            pie_data = pie_data[pie_data["Value"] > 0]
                            if not pie_data.empty:
                                fig = px.pie(pie_data, names="Category", values="Value", title="Pie Chart")
                                st.plotly_chart(fig, key=f"{prefix}_pie")
                            else:
                                st.warning("No positive numeric values available for Pie Chart visualization.")
                        except Exception as e:
                            st.error(f"❌ Failed to render Pie Chart: {str(e)}")
                            if st.session_state.debug_mode:
                                st.sidebar.error(f"Pie Chart Error Details: {str(e)}")
                    else:
                        st.warning("No data available for Pie Chart visualization.")
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_col, y=y_col, title=chart_type)
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_line")
                elif chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col, title=chart_type)
                    fig.update_layout(**axis_layout)
                    st.plotly_chart(fig, key=f"{prefix}_bar")
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
        """Toggle the About section visibility."""
        st.session_state.show_about = not st.session_state.show_about
        st.session_state.show_help = False
        st.session_state.show_history = False

    def toggle_help():
        """Toggle the Help section visibility."""
        st.session_state.show_help = not st.session_state.show_help
        st.session_state.show_about = False
        st.session_state.show_history = False

    def toggle_history():
        """Toggle the History section visibility."""
        st.session_state.show_history = not st.session_state.show_history
        st.session_state.show_about = False
        st.session_state.show_help = False

    # --- Sidebar Content ---
    with st.sidebar:
        # Custom CSS for sidebar buttons
        st.markdown("""
        <style>
        /* Default styling for sidebar buttons (suggested questions) */
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
        /* Custom styling for Clear conversation, About, Help & Documentation, and History buttons */
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

        # Display Snowflake logo
        logo_url = "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg"
        st.image(logo_url, width=250)

        # Button to clear conversation
        if st.button("Clear conversation", key="clear_conversation_button"):
            start_new_conversation()

        # Data source selection
        st.radio("Select Data Source:", ["Database", "Document"], key="data_source")

        # Configuration options
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

        # Sample questions section
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

        # History, About, Help sections
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

    # Display welcome message if no chat history exists
    if st.session_state.show_greeting and not st.session_state.chat_history:
        st.markdown("Welcome! I’m the Snowflake AI Assistant, ready to assist you with grant data analysis, summaries, and answers — simply type your question to get started.")
    else:
        st.session_state.show_greeting = False

    # Display chat history
    for message in st.session_state.chat_history:
        try:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)
                if message["role"] == "assistant" and "results" in message and message["results"] is not None:
                    with st.expander("View SQL Query", expanded=False):
                        st.code(message["sql"], language="sql")
                    st.markdown(f"**Query Results ({len(message['results'])}) rows:**")
                    # Display DataFrame without 'k' formatting for numerical columns
                    st.dataframe(
                        message["results"],
                        column_config={
                            col: st.column_config.NumberColumn(format="%d")  # Use plain integer format
                            for col in message["results"].columns
                            if pd.api.types.is_numeric_dtype(message["results"][col])
                        }
                    )
                    if not message["results"].empty and len(message["results"].columns) >= 2:
                        st.markdown("**📈 Visualization:**")
                        display_chart_tab(message["results"], prefix=f"chart_{hash(message['content'])}", query=message.get("query", ""))
        except Exception as e:
            st.error(f"Error displaying message: {str(e)}")

    # Get user input
    chat_input_query = st.chat_input("Ask your question...")
    if chat_input_query:
        st.session_state.query = chat_input_query

    # --- Process User Query ---
    if st.session_state.query:
        try:
            query = st.session_state.query
            if query.lower().startswith("no of"):
                query = query.replace("no of", "number of", 1)
            
            # Reset UI state
            st.session_state.show_greeting = False
            st.session_state.chart_x_axis = None
            st.session_state.chart_y_axis = None
            st.session_state.chart_type = "Bar Chart"
            original_query = query
            # Handle numeric input as a selection from last_suggestions
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
                        # Ensure data_source is valid
                        if st.session_state.data_source not in ["Database", "Document"]:
                            st.session_state.data_source = "Database"
                        # Determine query type
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

                        # Handle greeting query
                        if is_greeting or is_suggestion:
                            # Define suggested questions for the user
                            suggested_questions = [
                                "What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?",
                                "Give me date-wise award breakdowns.",
                                "What is this document about?",
                                "List all subject areas."
                            ]

                            # Format the response with the new greeting
                            response_content = (
                                "Hello! Welcome to the GRANTS AI Assistant!\n"
                                "I'm here to help you explore and analyze grant-related data, answer questions about awards, budgets, and more, or provide insights from documents.\n\n"
                                "Here are some questions you can try:\n\n" +
                                "\n".join([f"{i}. {question}" for i, question in enumerate(suggested_questions, 1)]) +
                                "\n\nFeel free to ask anything, or pick one of the suggested questions to get started!"
                            )

                            # Display the response
                            with response_placeholder:
                                st.markdown(response_content, unsafe_allow_html=True)

                            # Update session state
                            assistant_response["content"] = response_content
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response["content"]})
                            st.session_state.last_suggestions = suggested_questions

                        # Handle content generation queries
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

                        # Handle summary queries
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

                        # Handle structured queries (Database)
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
                                    st.markdown(f"**Query Results ({len(results)} rows):**")
                                    st.dataframe(
                                        results,
                                        column_config={
                                            col: st.column_config.NumberColumn(format="%d")
                                            for col in results.columns
                                            if pd.api.types.is_numeric_dtype(results[col])
                                        }
                                    )
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

                        # Handle document-based queries
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

                        # Provide suggestions if the query fails
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

                        # Update chat history and reset query
                        if assistant_response["content"]:
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
