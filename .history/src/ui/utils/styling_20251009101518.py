# UI Styling and Theme Management
import streamlit as st

def apply_custom_styling():
    """Apply custom styling to Streamlit app"""
    st.markdown("""
    <style>
        :root {
            --ev-colors-primary: #282828;
            --ev-colors-secondary: #424242;
            --ev-colors-tertiary: #4e332a;
            --ev-colors-highlight: #e75f33;
            --ev-colors-text: #fff;
            --ev-colors-secondaryText: grey;
            --ev-colors-tertiaryText: #a3a3a3;
            --ev-colors-borderColor: #ffffff1f;
            --ev-colors-background: #161616;
            --ev-colors-success: #d8fc77;
            --ev-colors-danger: #dc143c;
        }
        
        /* Main background */
        .stApp {
            background: var(--ev-colors-background);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: var(--ev-colors-primary);
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--ev-colors-text);
        }
        
        /* Text */
        .stMarkdown {
            color: var(--ev-colors-text);
        }
        
        /* Buttons */
        .stButton > button {
            background-color: var(--ev-colors-highlight);
            color: var(--ev-colors-text);
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #ff6b47;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(231, 95, 51, 0.3);
        }
        
        /* Primary buttons */
        .stButton > button[kind="primary"] {
            background-color: var(--ev-colors-highlight);
            color: var(--ev-colors-text);
        }
        
        /* Secondary buttons */
        .stButton > button[kind="secondary"] {
            background-color: var(--ev-colors-secondary);
            color: var(--ev-colors-text);
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            background-color: var(--ev-colors-secondary);
            color: var(--ev-colors-text);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 8px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--ev-colors-highlight);
            box-shadow: 0 0 0 2px rgba(231, 95, 51, 0.2);
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background-color: var(--ev-colors-secondary);
            color: var(--ev-colors-text);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 8px;
        }
        
        /* Sliders */
        .stSlider > div > div > div > div {
            background-color: var(--ev-colors-highlight);
        }
        
        /* Checkboxes */
        .stCheckbox > label {
            color: var(--ev-colors-text);
        }
        
        .stCheckbox > div > div {
            background-color: var(--ev-colors-secondary);
            border: 1px solid var(--ev-colors-borderColor);
        }
        
        /* Radio buttons */
        .stRadio > label {
            color: var(--ev-colors-text);
        }
        
        .stRadio > div > label {
            color: var(--ev-colors-text);
        }
        
        /* File uploader */
        .stFileUploader > div {
            background-color: var(--ev-colors-secondary);
            border: 2px dashed var(--ev-colors-borderColor);
            border-radius: 8px;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background-color: var(--ev-colors-highlight);
        }
        
        /* Success messages */
        .stSuccess {
            background-color: rgba(216, 252, 119, 0.1);
            border: 1px solid var(--ev-colors-success);
            color: var(--ev-colors-success);
        }
        
        /* Error messages */
        .stError {
            background-color: rgba(220, 20, 60, 0.1);
            border: 1px solid var(--ev-colors-danger);
            color: var(--ev-colors-danger);
        }
        
        /* Warning messages */
        .stWarning {
            background-color: rgba(255, 193, 7, 0.1);
            border: 1px solid #ffc107;
            color: #ffc107;
        }
        
        /* Info messages */
        .stInfo {
            background-color: rgba(13, 202, 240, 0.1);
            border: 1px solid #0dcaf0;
            color: #0dcaf0;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: var(--ev-colors-secondary);
            color: var(--ev-colors-text);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 8px;
        }
        
        .streamlit-expanderContent {
            background-color: var(--ev-colors-primary);
            color: var(--ev-colors-text);
            border: 1px solid var(--ev-colors-borderColor);
            border-top: none;
            border-radius: 0 0 8px 8px;
        }
        
        /* Tabs */
        .stTabs > div > div > div > div {
            background-color: var(--ev-colors-secondary);
            color: var(--ev-colors-text);
            border: 1px solid var(--ev-colors-borderColor);
        }
        
        .stTabs > div > div > div > div[aria-selected="true"] {
            background-color: var(--ev-colors-highlight);
            color: var(--ev-colors-text);
        }
        
        /* Columns */
        .stColumn {
            background-color: transparent;
        }
        
        /* Metrics */
        .stMetric {
            background-color: var(--ev-colors-secondary);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 8px;
            padding: 1rem;
        }
        
        .stMetric > div > div {
            color: var(--ev-colors-text);
        }
        
        /* Dataframes */
        .stDataFrame {
            background-color: var(--ev-colors-secondary);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 8px;
        }
        
        /* Code blocks */
        .stCode {
            background-color: var(--ev-colors-primary);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 8px;
        }
        
        /* JSON */
        .stJson {
            background-color: var(--ev-colors-primary);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 8px;
        }
        
        /* Remove Streamlit default colors */
        .css-1d391kg {
            background-color: var(--ev-colors-primary) !important;
        }
        
        .css-1d391kg .css-1d391kg {
            background-color: var(--ev-colors-primary) !important;
        }
        
        /* Remove Streamlit default highlights */
        .css-1d391kg .css-1d391kg .css-1d391kg {
            background-color: var(--ev-colors-primary) !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--ev-colors-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--ev-colors-highlight);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #ff6b47;
        }
        
        /* Loading spinner */
        .stSpinner {
            color: var(--ev-colors-highlight);
        }
        
        /* Tooltips */
        .stTooltip {
            background-color: var(--ev-colors-primary);
            color: var(--ev-colors-text);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 8px;
        }
        
        /* Footer */
        .stFooter {
            background-color: var(--ev-colors-primary);
            color: var(--ev-colors-text);
            border-top: 1px solid var(--ev-colors-borderColor);
        }
        
        /* Custom classes */
        .custom-card {
            background-color: var(--ev-colors-secondary);
            border: 1px solid var(--ev-colors-borderColor);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .custom-card:hover {
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }
        
        .status-pending {
            color: #ffc107;
        }
        
        .status-processing {
            color: #0dcaf0;
        }
        
        .status-completed {
            color: var(--ev-colors-success);
        }
        
        .status-error {
            color: var(--ev-colors-danger);
        }
        
        .highlight-text {
            color: var(--ev-colors-highlight);
            font-weight: 600;
        }
        
        .muted-text {
            color: var(--ev-colors-tertiaryText);
        }
        
        .success-text {
            color: var(--ev-colors-success);
        }
        
        .error-text {
            color: var(--ev-colors-danger);
        }
    </style>
    """, unsafe_allow_html=True)

def get_logo_svg():
    """Get the logo SVG"""
    return """<svg id="Layer_2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1703.31 535.6"><defs><style>                                                
        .cls-1{fill:#fff;}.cls-2{fill:#e75f33;}.cls-3{fill:#d8fc77;}
    </style></defs><g id="Layer_10"><g><path class="cls-1" d="M125.67,428.34c-39.15,0-70.27-13.09-92.48-38.91C11.17,363.84,0,334.47,0,302.15c0-30.4,9.47-57.88,28.14-81.68,23.77-30.39,56.01-45.8,95.83-45.8s74.1,15.76,98.58,46.85c17.39,21.95,26.36,49.63,26.66,82.28l.05,5.23H41.22c1.5,23.04,9.58,42.3,24.08,57.31,15.74,16.28,34.65,24.2,57.81,24.2,11.12,0,22.08-1.96,32.6-5.83,10.49-3.85,19.51-9.02,26.82-15.36,7.36-6.39,8.83-7.95,14.56-15.39l2.6-4.32c5.42-9.02,16.94-12.25,26.26-7.35h0c9.62,5.06,13.39,16.91,8.46,26.6l-1.53,3c-8.02,11.54-10.34,14.39-21.53,24.68-11.22,10.32-24.02,18.29-38.05,23.68-14.02,5.38-30.04,8.1-47.63,8.1ZM204.47,272.93c-3.65-12.13-8.55-22.08-14.6-29.64-7.06-8.82-16.57-16.06-28.27-21.51-11.75-5.46-24.27-8.23-37.2-8.23-21.29,0-39.83,6.92-55.1,20.58-9.88,8.81-17.76,21.84-23.46,38.8h158.64Z"></path><rect class="cls-1" x="288.28" y="97.26" width="40.15" height="331.08" rx="20.07" ry="20.07"></rect><path class="cls-1" d="M490.58,428.34c-39.15,0-70.27-13.09-92.48-38.91-22.02-25.59-33.19-54.96-33.19-87.28,0-30.4,9.47-57.88,28.14-81.68,23.77-30.39,56.01-45.8,95.83-45.8s74.1,15.76,98.58,46.85c17.39,21.95,26.36,49.63,26.66,82.28l.05,5.23h-208.03c1.5,23.04,9.58,42.3,24.08,57.31,15.74,16.28,34.65,24.2,57.81,24.2,11.12,0,22.08-1.96,32.6-5.83,10.49-3.85,19.51-9.02,26.82-15.36,7.36-6.39,8.83-7.95,14.56-15.39l2.6-4.32c5.42-9.02,16.94-12.25,26.26-7.35h0c9.62,5.06,13.39,16.91,8.46,26.6l-1.53,3c-8.02,11.54-10.34,14.39-21.53,24.68-11.22,10.32-24.02,18.29-38.05,23.68-14.02,5.38-30.04,8.1-47.63,8.1ZM569.37,272.93c-3.65-12.13-8.55-22.08-14.6-29.64-7.06-8.82-16.57-16.06-28.27-21.51-11.75-5.46-24.27-8.23-37.2-8.23-21.29,0-39.83,6.92-55.1,20.58-9.89,8.81-17.76,21.84-23.46,38.8h158.64Z"></path><path class="cls-1" d="M751.92,422.82l-96-208.47c-5.97-12.97,3.5-27.77,17.78-27.77h0c7.64,0,14.59,4.45,17.78,11.39l69.08,150.01,68.21-149.93c3.18-6.99,10.15-11.47,17.82-11.47h.22c14.26,0,23.74,14.76,17.8,27.73l-95.43,208.49c-1.55,3.38-4.92,5.54-8.63,5.54h0c-3.71,0-7.08-2.16-8.63-5.52Z"></path><g><path class="cls-2" d="M1052.79,311.55c-30.67,0-56.25,33.01-62.14,66.95,5.07-11.19,11.63-17.94,18.79-17.94,15.94,0,23.38,33.67,28.84,74.37,1.51,11.28,12.67,86.53,13.56,100.67.05,0,.11,0,.16,0,1.04-16.27,10.83-87.61,12.64-100.66,5.78-41.56,12.93-74.37,28.87-74.37,9.09,0,17.21,10.84,22.5,27.76-2.22-38.69-29.66-76.77-63.22-76.77Z"></path><path class="cls-3" d="M1053.33,46.78c60,50.38,96.73,131.67,97.74,218.86-26.55-32.52-60.86-50.27-97.76-50.27s-71.19,17.74-97.74,50.24c1.01-87.19,37.75-168.47,97.75-218.83M1053.33,0c-80.86,53.76-135.27,154.25-135.27,269.32,0,28.59,3.36,56.29,9.66,82.6,4.47,18.64,10.39,36.6,17.66,53.67,2.54-84.98,49.89-152.72,107.94-152.72s105.41,67.76,107.94,152.76c10.02-23.52,17.51-48.73,22.09-75.13,3.46-19.78,5.25-40.25,5.25-61.19C1188.59,154.25,1134.19,53.78,1053.33,0h0Z"></path></g><path class="cls-3" d="M1246.12,390.85l-15.96-370.06C1229.55,9.49,1238.55,0,1249.87,0h0c11.31,0,20.31,9.49,19.71,20.79l-15.96,370.06h-7.5Z"></path><path class="cls-1" d="M1333.96,408.27v-185.58h-40.62v-36.1h40.62v-69.25c0-11.09,8.99-20.07,20.07-20.07h0c11.09,0,20.07,8.99,20.07,20.07v69.25h62.21v36.1h-62.21v185.58c0,11.09-8.99,20.07-20.07,20.07h0c-11.09,0-20.07-8.99-20.07-20.07Z"></path><path class="cls-1" d="M1579.72,428.34c-39.15,0-70.26-13.09-92.48-38.91-22.02-25.59-33.18-54.95-33.18-87.28,0-30.4,9.47-57.88,28.14-81.68,23.77-30.39,56.01-45.8,95.83-45.8s74.1,15.76,98.59,46.85c17.39,21.94,26.36,49.63,26.66,82.28l.05,5.23h-208.03c1.5,23.04,9.59,42.3,24.08,57.31,15.74,16.28,34.64,24.2,57.81,24.2,11.12,0,22.09-1.96,32.6-5.83,10.49-3.85,19.51-9.02,26.82-15.36,7.36-6.39,9.22-7.53,15.54-17.02l1.62-2.69c5.42-9.02,16.94-12.25,26.26-7.35h0c9.62,5.06,13.39,16.91,8.46,26.6l-1.36,2.67c-6.09,8.44-10.51,14.72-21.7,25.01-11.22,10.32-24.02,18.29-38.06,23.68-14.02,5.38-30.04,8.1-47.63,8.1ZM1658.52,272.93c-3.65-12.13-8.55-22.08-14.6-29.64-7.06-8.82-16.57-16.06-28.27-21.51-11.76-5.46-24.27-8.23-37.2-8.23-21.29,0-39.83,6.92-55.1,20.58-9.89,8.81-17.76,21.85-23.46,38.8h158.64Z"></path></g></g></svg>"""

def render_logo():
    """Render the logo in the sidebar"""
    st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        {get_logo_svg()}
    </div>
    """, unsafe_allow_html=True)

def get_status_color(status: str) -> str:
    """Get color for status"""
    colors = {
        "pending": "#ffc107",
        "processing": "#0dcaf0", 
        "completed": "#d8fc77",
        "error": "#dc143c",
        "ready": "#d8fc77"
    }
    return colors.get(status, "#6c757d")

def get_status_icon(status: str) -> str:
    """Get icon for status"""
    icons = {
        "pending": "⏳",
        "processing": "🔄",
        "completed": "✅",
        "error": "❌",
        "ready": "🎯"
    }
    return icons.get(status, "❓")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def is_large_file(file_size: int, threshold_mb: int = 10) -> bool:
    """Check if file is considered large"""
    return file_size > threshold_mb * 1024 * 1024

def create_progress_bar(status_dict: dict) -> str:
    """Create HTML progress bar"""
    total_steps = len(status_dict)
    completed_steps = sum(1 for status in status_dict.values() if status in ["completed", "ready"])
    progress_percent = (completed_steps / total_steps) * 100
    
    html = f"""
    <div style="background-color: #424242; border-radius: 10px; padding: 4px; margin: 10px 0;">
        <div style="background-color: #e75f33; height: 20px; border-radius: 6px; width: {progress_percent}%; transition: width 0.3s ease;"></div>
    </div>
    <div style="text-align: center; color: #fff; font-size: 14px; margin-top: 5px;">
        {completed_steps}/{total_steps} steps completed ({progress_percent:.0f}%)
    </div>
    """
    return html

def create_status_badge(status: str, text: str = None) -> str:
    """Create status badge"""
    if text is None:
        text = status.title()
    
    color = get_status_color(status)
    icon = get_status_icon(status)
    
    return f"""
    <span style="
        background-color: {color}20;
        color: {color};
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid {color};
        display: inline-flex;
        align-items: center;
        gap: 4px;
    ">
        {icon} {text}
    </span>
    """

def create_info_card(title: str, content: str, icon: str = "ℹ️") -> str:
    """Create info card"""
    return f"""
    <div class="custom-card">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 20px; margin-right: 8px;">{icon}</span>
            <h4 style="margin: 0; color: #e75f33;">{title}</h4>
        </div>
        <p style="margin: 0; color: #fff;">{content}</p>
    </div>
    """

def create_metric_card(label: str, value: str, delta: str = None) -> str:
    """Create metric card"""
    delta_html = ""
    if delta:
        delta_color = "#d8fc77" if delta.startswith("+") else "#dc143c"
        delta_html = f'<div style="color: {delta_color}; font-size: 14px; margin-top: 4px;">{delta}</div>'
    
    return f"""
    <div class="custom-card" style="text-align: center;">
        <div style="color: #a3a3a3; font-size: 14px; margin-bottom: 8px;">{label}</div>
        <div style="color: #fff; font-size: 24px; font-weight: 600;">{value}</div>
        {delta_html}
    </div>
    """