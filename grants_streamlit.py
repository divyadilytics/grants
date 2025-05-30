import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px

# Snowflake/Cortex Configuration
HOST = "GBJYVCT-LSB50763.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.Grants_search_services"

# Single semantic model - Commented out since the file is missing
# SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/GRANTSyaml_27.yaml'

# Streamlit Page Config
st.set_page_config(
    page_title="Welcome to Cortex AI Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
    st.session_state.chat_history = []
    st.session_state.data_source = "Database"  # Initialize data source
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = ""
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = ""
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Pie"
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested_question_selected" not in st.session_state:
    st.session_state.suggested_question_selected = None

# Hide Streamlit branding and prevent chat history shading
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Function to start a new conversation
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
    st.session_state.suggested_question_selected = None
    st.rerun()

# Authentication logic
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
            st.error(f"Authentication failed: {e}")
else:
    session = st.session_state.snowpark_session

    # Utility Functions
    def run_snowflake_query(query):
        try:
            if not query:
                st.warning("⚠️ No SQL query generated.")
                return None
            df = session.sql(query)
            data = df.collect()
            if not data:
                return None
            columns = df.schema.names
            result_df = pd.DataFrame(data, columns=columns)
            return result_df
        except Exception as e:
            st.error(f"❌ SQL Execution Error: {str(e)}")
            return None

    def is_structured_query(query: str):
        structured_patterns = [
            r'\b(county|number|where|group by|order by|completed units|sum|count|avg|max|min|least|highest|which)\b',
            r'\b(total|how many|leads|profit|projects|jurisdiction|month|year|energy savings|kwh)\b'
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
            r'^\b(hello|hi)\b$',
            r'^\b(hello|hi)\b\s.*$'
        ]
        return any(re.search(pattern, query.lower()) for pattern in greeting_patterns)

    def complete(prompt, model="mistral-large"):
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

    def snowflake_api_call(query: str, is_structured: bool = False):
        payload = {
            "model": "mistral-large",
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if st.session_state.data_source == "Document" or is_structured:
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 1}}
        try:
            resp = requests.post(
                url=f"https://{HOST}{API_ENDPOINT}",
                json=payload,
                headers={
                    "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
                    "Content-Type": "application/json",
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
            st.error(f"❌ API Request Failed: {str(e)}")
            return None

    def summarize_unstructured_answer(answer):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|")\s', answer)
        return "\n".join(f"• {sent.strip()}" for sent in sentences[:6])

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

    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart", query: str = ""):
        if df.empty or len(df.columns) < 2:
            return
        query_lower = query.lower()
        if re.search(r'\b(county|jurisdiction)\b', query_lower):
            default_chart = "Pie Chart"
        elif re.search(r'\b(month|year|date)\b', query_lower):
            default_chart = "Line Chart"
        else:
            default_chart = "Bar Chart"
        all_cols = list(df.columns)
        col1, col2, col3 = st.columns(3)
        default_x = st.session_state.get(f"{prefix}_x", all_cols[0])
        try:
            x_index = all_cols.index(default_x)
        except ValueError:
            x_index = 0
        x_col = col1.selectbox("X axis", all_cols, index=x_index, key=f"{prefix}_x")
        remaining_cols = [c for c in all_cols if c != x_col]
        default_y = st.session_state.get(f"{prefix}_y", remaining_cols[0])
        try:
            y_index = remaining_cols.index(default_y)
        except ValueError:
            y_index = 0
        y_col = col2.selectbox("Y axis", remaining_cols, index=y_index, key=f"{prefix}_y")
        chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
        default_type = st.session_state.get(f"{prefix}_type", default_chart)
        try:
            type_index = chart_options.index(default_type)
        except ValueError:
            type_index = chart_options.index(default_chart)
        chart_type = col3.selectbox("Chart Type", chart_options, index=type_index, key=f"{prefix}_type")
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_line")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_bar")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_pie")
        elif chart_type == "Scatter Chart":
            fig = px.scatter(df, x=x_col, y=y_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_scatter")
        elif chart_type == "Histogram Chart":
            fig = px.histogram(df, x=x_col, title=chart_type)
            st.plotly_chart(fig, key=f"{prefix}_hist")

    # UI Logic
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
        </style>
        """, unsafe_allow_html=True)

        logo_container = st.container()
        button_container = st.container()
        data_source_container = st.container()
        about_container = st.container()
        help_container = st.container()

        with logo_container:
            logo_url = "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg"
            st.image(logo_url, width=250)

        with button_container:
            st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
            if st.button("New Conversation", key="new_conversation"):
                start_new_conversation()

        with data_source_container:
            # Use st.radio and assign its value to session state
            data_source = st.radio(
                "Select Data Source:",
                ["Database", "Document"],
                index=0 if st.session_state.data_source == "Database" else 1,
                key="data_source",
                help="Choose 'Database' for structured queries or 'Document' for document-based queries."
            )
            # Update session state only if the value changes
            if data_source != st.session_state.data_source:
                st.session_state.data_source = data_source

        with about_container:
            st.markdown("### About")
            st.write(
                "This application uses **Snowflake Cortex Analyst** to interpret "
                "your natural language questions and generate data insights. "
                "Simply ask a question below to see relevant answers and visualizations."
            )

        with help_container:
            st.markdown("### Help & Documentation")
            st.write(
                "- [User Guide](https://docs.snowflake.com/en/guides-overview-ai-features)  \n"
                "- [Snowflake Cortex Analyst Docs](https://docs.snowflake.com/)  \n"
                "- [Contact Support](https://www.snowflake.com/en/support/)"
            )

    st.title("Cortex AI Assistant by DiLytics")

    # Semantic model display is commented out since the file is missing
    # semantic_model_filename = SEMANTIC_MODEL.split("/")[-1]
    # st.markdown(f"Semantic Model: `{semantic_model_filename}`")

    # Initialize service metadata (assuming this function is defined elsewhere)
    init_service_metadata()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "results" in message and message["results"] is not None:
                with st.expander("View SQL Query", expanded=False):
                    st.code(message["sql"], language="sql")
                st.markdown(f"**Query Results ({len(message['results'])} rows):**")
                st.dataframe(message["results"])
                if not message["results"].empty and len(message["results"].columns) >= 2:
                    st.markdown("**📈 Visualization:**")
                    display_chart_tab(message["results"], prefix=f"chart_{hash(message['content'])}", query=message.get("query", ""))

            if message["role"] == "assistant" and not message.get("understood_query", True):
                st.markdown("**I’m sorry, I didn’t understand your question. Here are some suggested questions you can try:**")
                for idx, suggestion in enumerate(sample_questions):
                    if st.button(suggestion, key=f"suggestion_{hash(message['query'])}_{idx}"):
                        st.session_state.suggested_question_selected = suggestion
                        st.rerun()

    # Get user query from chat input or sidebar sample questions
    query = st.chat_input("Ask your question...")

    sample_questions = [
        "what is the total actual award budget?",
        "What is the total actual award posted",
        "What is the total amount of award encumbrances approved",
        "What is the total task actual posted by award name?"
    ]

    for sample in sample_questions:
        if st.sidebar.button(sample, key=sample):
            query = sample

    if st.session_state.suggested_question_selected:
        query = st.session_state.suggested_question_selected
        st.session_state.suggested_question_selected = None

    if query:
        st.session_state.chart_x_axis = None
        st.session_state.chart_y_axis = None
        st.session_state.chart_type = "Bar Chart"
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Generating Response..."):
                is_structured = is_structured_query(query) and st.session_state.data_source == "Database"
                is_complete = is_complete_query(query)
                is_summarize = is_summarize_query(query)
                is_suggestion = is_question_suggestion_query(query)
                is_greeting = is_greeting_query(query)
                understood_query = False
                assistant_response = {"role": "assistant", "content": "", "query": query}

                if is_greeting:
                    response_content = (
                        "Hello! Welcome to the GRANTS AI Assistant! I'm here to help you explore and analyze "
                        f"{'grant-related data' if st.session_state.data_source == 'Database' else 'documents'}."
                        "\n\nHere are some questions you can try:\n"
                        "- What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?\n"
                        "- Give me date-wise award breakdowns.\n"
                        "- What is this document about?\n"
                        "- List all subject areas.\n\n"
                        "Feel free to ask anything, or pick one of the suggested questions to get started!"
                    )
                    st.markdown(response_content)
                    assistant_response["content"] = response_content
                    understood_query = True

                elif is_suggestion:
                    response_content = "**Here are some questions you can ask me:**\n"
                    for i, q in enumerate(sample_questions, 1):
                        response_content += f"{i}. {q}\n"
                    response_content += f"\nFeel free to ask any of these or come up with your own related to {'energy savings, Green Residences, or other programs' if st.session_state.data_source == 'Database' else 'documents'}!"
                    st.markdown(response_content)
                    assistant_response["content"] = response_content
                    understood_query = True

                elif is_complete:
                    response = complete(query)
                    if response:
                        response_content = f"**✍️ Generated Response:**\n{response}"
                        st.markdown(response_content)
                        assistant_response["content"] = response_content
                        understood_query = True
                    else:
                        response_content = "⚠️ Failed to generate a response."
                        st.warning(response_content)
                        assistant_response["content"] = response_content

                elif is_summarize:
                    summary = summarize(query)
                    if summary:
                        response_content = f"**Summary:**\n{summary}"
                        st.markdown(response_content)
                        assistant_response["content"] = response_content
                        understood_query = True
                    else:
                        response_content = "⚠️ Failed to generate a summary."
                        st.warning(response_content)
                        assistant_response["content"] = response_content

                elif is_structured:
                    response = snowflake_api_call(query, is_structured=True)
                    sql, _ = process_sse_response(response, is_structured=True)
                    if sql:
                        results = run_snowflake_query(sql)
                        if results is not None and not results.empty:
                            results_text = results.to_string(index=False)
                            prompt = f"Provide a concise natural language answer to the query '{query}' using the following data, avoiding phrases like 'Based on the query results':\n\n{results_text}"
                            summary = complete(prompt)
                            if not summary:
                                summary = "⚠️ Unable to generate a natural language summary."
                            response_content = f"**✍️ Generated Response:**\n{summary}"
                            st.markdown(response_content)
                            with st.expander("View SQL Query", expanded=False):
                                st.code(sql, language="sql")
                            st.markdown(f"**Query Results ({len(results)} rows):**")
                            st.dataframe(results)
                            if len(results.columns) >= 2:
                                st.markdown("**📈 Visualization:**")
                                display_chart_tab(results, prefix=f"chart_{hash(query)}", query=query)
                            assistant_response.update({
                                "content": response_content,
                                "sql": sql,
                                "results": results,
                                "summary": summary
                            })
                            understood_query = True
                        else:
                            response_content = "⚠️ No data found."
                            st.warning(response_content)
                            assistant_response["content"] = response_content
                    else:
                        response_content = "⚠️ No SQL generated."
                        st.warning(response_content)
                        assistant_response["content"] = response_content

                else:
                    response = snowflake_api_call(query, is_structured=False)
                    _, search_results = process_sse_response(response, is_structured=False)
                    if search_results:
                        raw_result = search_results[0]
                        summary = summarize(raw_result)
                        if summary:
                            response_content = f"**Here is the Answer:**\n{summary}"
                            last_sentence = summary.split(".")[-2] if "." in summary else summary
                            st.markdown(response_content)
                            st.success(f"Key Insight: {last_sentence.strip()}")
                            assistant_response["content"] = response_content
                            understood_query = True
                        else:
                            response_content = f"**🔍 Key Information (Unsummarized):**\n{summarize_unstructured_answer(raw_result)}"
                            st.markdown(response_content)
                            assistant_response["content"] = response_content
                            understood_query = True
                    else:
                        response_content = "⚠️ No relevant search results found."
                        st.warning(response_content)
                        assistant_response["content"] = response_content

                assistant_response["understood_query"] = understood_query
                st.session_state.chat_history.append(assistant_response)
                st.session_state.messages.append(assistant_response)
                st.session_state.current_query = query
                st.session_state.current_results = assistant_response.get("results")
                st.session_state.current_sql = assistant_response.get("sql")
                st.session_state.current_summary = assistant_response.get("summary")

    if not st.session_state.messages:
        st.markdown("💡 **Welcome! I’m the Snowflake Cortex AI Assistant, ready to assist you with grant data analysis — simply type your question to get started**")

