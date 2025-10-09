# UI Styling and Themes
import streamlit as st

def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit app"""
    
    # Custom CSS for dark theme and better styling
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #FF8C00;
        --secondary-color: #FFA500;
        --background-color: #0E1117;
        --text-color: #FAFAFA;
        --card-background: #262730;
        --border-color: #3E3E3E;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(45deg, #FF8C00, #FFA500);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    /* Process tracker styling */
    .process-tracker {
        background: linear-gradient(45deg, #FF8C00, #FFA500);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .process-tracker h2 {
        color: white;
        text-align: center;
        margin: 0;
    }
    
    /* Status indicators */
    .status-pending {
        color: #FFA500;
        font-weight: bold;
    }
    
    .status-completed {
        color: #00FF00;
        font-weight: bold;
    }
    
    .status-ready {
        color: #00BFFF;
        font-weight: bold;
    }
    
    .status-error {
        color: #FF0000;
        font-weight: bold;
    }
    
    /* Card styling */
    .info-card {
        background: var(--card-background);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF8C00, #FFA500);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 140, 0, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--card-background);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #FF8C00, #FFA500);
    }
    
    /* Metric styling */
    .metric-container {
        background: var(--card-background);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-color);
        opacity: 0.8;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-color);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(45deg, #FF8C00, #FFA500);
    }
    
    /* Checkbox styling */
    .stCheckbox > div > label > div {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }
    
    /* Radio button styling */
    .stRadio > div > label > div {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs > div > div > div > div {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.9rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_status_indicator(status: str, text: str) -> str:
    """Render status indicator with appropriate styling"""
    status_colors = {
        "pending": "#FFA500",
        "completed": "#00FF00", 
        "ready": "#00BFFF",
        "error": "#FF0000"
    }
    
    color = status_colors.get(status, "#FFA500")
    
    return f"""
    <div style="display: flex; align-items: center; margin: 5px 0;">
        <div style="width: 12px; height: 12px; background-color: {color}; border-radius: 50%; margin-right: 10px;"></div>
        <span style="color: {color}; font-weight: bold;">{text}</span>
    </div>
    """

def render_metric_card(title: str, value: str, subtitle: str = "") -> str:
    """Render a metric card with custom styling"""
    return f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {f'<div style="font-size: 0.8rem; color: var(--text-color); opacity: 0.6; margin-top: 5px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """

def render_info_card(title: str, content: str) -> str:
    """Render an info card with custom styling"""
    return f"""
    <div class="info-card">
        <h4 style="color: var(--primary-color); margin: 0 0 10px 0;">{title}</h4>
        <div style="color: var(--text-color);">{content}</div>
    </div>
    """
