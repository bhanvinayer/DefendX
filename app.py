"""
DefendX Multi-Agent Fraud Detection System
Main application entry point for Streamlit Cloud deployment
"""

import sys
import os

# Add src directory to Python path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Now import and run the main application
try:
    from app import main
    
    if __name__ == "__main__":
        main()
except Exception as e:
    import streamlit as st
    st.error(f"Failed to import main application: {e}")
    st.info("Please check that all dependencies are properly installed.")