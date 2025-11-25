"""
Streamlit Web Interface for Customer Churn Prediction Chatbot.

This app provides an interactive chat interface with visualizations
for analyzing customer churn using AI-powered insights.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from agents.churn_agent import ChurnAgent, get_agent
from database.queries import get_churn_rate, ChurnDatabase
from models.predictor import ChurnPredictor
from config import validate_paths, OPENAI_API_KEY


# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="Customer Churn Prediction Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== CUSTOM CSS ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #000000;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
        color: #1a1a1a;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
        color: #1a1a1a;
    }
    .tool-call {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        font-family: monospace;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ==================== INITIALIZATION ====================

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = get_agent()
            st.session_state.agent_initialized = True
        except Exception as e:
            st.session_state.agent = None
            st.session_state.agent_initialized = False
            st.session_state.init_error = str(e)
    
    if 'show_reasoning' not in st.session_state:
        st.session_state.show_reasoning = False
    
    if 'stats_loaded' not in st.session_state:
        st.session_state.stats_loaded = False


def check_configuration():
    """Check if system is properly configured."""
    issues = validate_paths()
    
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        issues.append("OpenAI API key not configured in .env file")
    
    return issues


# ==================== SIDEBAR ====================

def render_sidebar():
    """Render sidebar with stats and controls."""
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/000000/chatbot.png", width=100)
        st.title("🤖 Churn Assistant")
        
        # System status
        st.subheader("📊 System Status")
        
        issues = check_configuration()
        if issues:
            st.error("⚠️ Configuration Issues:")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("✅ System Ready")
        
        if st.session_state.agent_initialized:
            agent_info = st.session_state.agent.get_agent_info()
            st.info(f"""
            **Model:** {agent_info['model']}  
            **Tools:** {agent_info['num_tools']}  
            **Conversation:** {agent_info['conversation_length']} messages
            """)
        
        st.divider()
        
        # Database Statistics
        st.subheader("📈 Quick Stats")
        
        try:
            stats = get_churn_rate()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Customers", f"{stats['total_customers']:,}")
                st.metric("Churned", f"{stats['churned']:,}", 
                         delta=f"-{stats['churn_rate']*100:.1f}%", delta_color="inverse")
            
            with col2:
                st.metric("Retained", f"{stats['retained']:,}",
                         delta=f"+{(1-stats['churn_rate'])*100:.1f}%")
                st.metric("Churn Rate", stats['churn_rate_pct'], delta_color="inverse")
            
            # Churn rate gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stats['churn_rate'] * 100,
                title={'text': "Churn Rate %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 40], 'color': "yellow"},
                        {'range': [40, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': stats['churn_rate'] * 100
                    }
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading stats: {e}")
        
        st.divider()
        
        # Controls
        st.subheader("⚙️ Settings")
        
        st.session_state.show_reasoning = st.checkbox(
            "Show Reasoning Steps",
            value=st.session_state.show_reasoning,
            help="Display the tools and steps the agent uses"
        )
        
        if st.button("🔄 Reset Conversation", use_container_width=True):
            if st.session_state.agent:
                st.session_state.agent.reset_memory()
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.show_analytics = True
        
        st.divider()
        
        # Quick Actions - NEW!
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💰 Financial Risk", use_container_width=True, help="Calculate revenue at risk from high-risk customers"):
                st.session_state.quick_action = "What's the financial impact of high-risk customers?"
                st.rerun()
            
            if st.button("🔍 Pattern Analysis", use_container_width=True, help="Analyze churn patterns and get recommendations"):
                st.session_state.quick_action = "Analyze high-risk customer patterns and give me recommendations"
                st.rerun()
        
        with col2:
            if st.button("⚔️ Compare Groups", use_container_width=True, help="Compare retained vs high-risk customers"):
                st.session_state.quick_action = "Compare top 5 retained vs top 5 high risk customers"
                st.rerun()
            
            if st.button("📊 Top High-Risk", use_container_width=True, help="Show top high-risk customers"):
                st.session_state.quick_action = "Show me top 5 high-risk customers"
                st.rerun()
        
        st.divider()
        
        # Example queries
        st.subheader("💡 Try These Queries")
        examples = [
            "What's the overall churn rate?",
            "Show churn for people without landline",
            "Find high-risk customers",
            "Why do customers churn?",
            "Predict churn for customer 7590-VHVEG"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}", use_container_width=True):
                st.session_state.example_query = example


# ==================== MAIN CHAT INTERFACE ====================

def render_chat_message(role: str, content: str, timestamp: str = None):
    """Render a chat message."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")
    
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.title()}</strong> <small style="color: gray;">({timestamp})</small>
        <div style="margin-top: 0.5rem;">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_tool_steps(intermediate_steps):
    """Render intermediate reasoning steps."""
    if not intermediate_steps:
        return
    
    with st.expander("🔍 View Reasoning Steps", expanded=False):
        for i, (action, observation) in enumerate(intermediate_steps, 1):
            tool_name = action.tool if hasattr(action, 'tool') else "Unknown Tool"
            tool_input = action.tool_input if hasattr(action, 'tool_input') else {}
            
            st.markdown(f"""
            <div class="tool-call">
                <strong>Step {i}: {tool_name}</strong><br>
                Input: {tool_input}<br>
                Result: {observation[:200]}{'...' if len(str(observation)) > 200 else ''}
            </div>
            """, unsafe_allow_html=True)


def render_chat_interface():
    """Render main chat interface."""
    st.markdown('<div class="main-header">🤖 Customer Churn Prediction Assistant</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Ask me anything about customer churn! I can:
    - 📊 Analyze churn rates and statistics
    - 🔮 Predict churn for specific customers
    - 🎯 Identify high-risk customers
    - 💡 Provide retention insights
    """)
    
    # Check if agent is initialized
    if not st.session_state.agent_initialized:
        st.error(f"""
        ⚠️ Agent initialization failed: {st.session_state.get('init_error', 'Unknown error')}
        
        Please check:
        1. OpenAI API key is set in .env file
        2. Model and pipeline files exist
        3. Database is accessible
        """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        render_chat_message(
            message['role'],
            message['content'],
            message.get('timestamp')
        )
        
        if st.session_state.show_reasoning and 'steps' in message:
            render_tool_steps(message['steps'])
    
    # Handle quick action button clicks FIRST (before chat input)
    if 'quick_action' in st.session_state:
        query = st.session_state.quick_action
        del st.session_state.quick_action  # Delete BEFORE processing
        process_user_input(query)
    
    # Handle example query button clicks
    elif 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query  # Delete BEFORE processing
        process_user_input(query)
    
    # Chat input (use elif to prevent multiple triggers)
    elif prompt := st.chat_input("Ask about customer churn..."):
        process_user_input(prompt)


def process_user_input(user_input: str):
    """Process user input and get agent response."""
    timestamp = datetime.now().strftime("%H:%M")
    
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input,
        'timestamp': timestamp
    })
    
    # Get agent response
    with st.spinner("🤔 Thinking..."):
        try:
            result = st.session_state.agent.query(user_input)
            
            response_msg = {
                'role': 'assistant',
                'content': result['output'],
                'timestamp': datetime.now().strftime("%H:%M")
            }
            
            if st.session_state.show_reasoning:
                response_msg['steps'] = result.get('intermediate_steps', [])
            
            st.session_state.messages.append(response_msg)
            
        except Exception as e:
            error_msg = {
                'role': 'assistant',
                'content': f"❌ Sorry, I encountered an error: {str(e)}",
                'timestamp': datetime.now().strftime("%H:%M")
            }
            st.session_state.messages.append(error_msg)
    
    st.rerun()


# ==================== ANALYTICS DASHBOARD ====================

def render_analytics():
    """Render analytics dashboard."""
    st.title("📊 Churn Analytics Dashboard")
    
    try:
        db = ChurnDatabase()
        
        # Get all customers
        customers = db.get_all_customers()
        
        # Churn distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn Distribution")
            churn_counts = customers['Churn'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=['Retained', 'Churned'] if churn_counts.index[0] == 'No' else ['Churned', 'Retained'],
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Churn by Contract Type")
            contract_churn = customers.groupby('Contract')['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values(ascending=False)
            
            fig = px.bar(
                x=contract_churn.index,
                y=contract_churn.values,
                labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
                color=contract_churn.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, width='stretch')
        
        # Additional charts
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Churn by Internet Service")
            internet_churn = customers.groupby('InternetService')['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            )
            fig = px.bar(x=internet_churn.index, y=internet_churn.values,
                        labels={'x': 'Internet Service', 'y': 'Churn Rate (%)'})
            st.plotly_chart(fig, width='stretch')
        
        with col4:
            st.subheader("Tenure vs Churn")
            # Convert tenure to numeric
            customers_numeric = customers.copy()
            customers_numeric['tenure'] = pd.to_numeric(customers_numeric['tenure'], errors='coerce')
            
            fig = px.histogram(
                customers_numeric,
                x='tenure',
                color='Churn',
                nbins=30,
                labels={'tenure': 'Tenure (months)', 'count': 'Number of Customers'},
                color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
            )
            st.plotly_chart(fig, width='stretch')
        
        # High-risk customers
        st.subheader("🔴 High-Risk Customers")
        with st.spinner("Running predictions..."):
            predictor = ChurnPredictor()
            high_risk = predictor.get_high_risk_customers(threshold=0.7, limit=20)
            
            if high_risk and 'error' not in high_risk[0]:
                df_risk = pd.DataFrame(high_risk)
                df_risk['churn_probability'] = df_risk['churn_probability'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    df_risk[['customer_id', 'churn_probability', 'risk_level']],
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info("No high-risk customers found")
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")


# ==================== MAIN APP ====================

def main():
    """Main application entry point."""
    initialize_session_state()
    
    render_sidebar()
    
    # Check if analytics view is requested
    if st.session_state.get('show_analytics', False):
        render_analytics()
        if st.button("← Back to Chat"):
            st.session_state.show_analytics = False
            st.rerun()
    else:
        render_chat_interface()


if __name__ == "__main__":
    main()
