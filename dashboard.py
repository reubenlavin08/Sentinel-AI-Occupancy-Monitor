import streamlit as st
import sqlite3
import pandas as pd
import time
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sentinel Occupancy Dashboard", layout="wide")
st.title("Live Camera Occupancy Analytics")

# --- DATABASE CONNECTION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, 'occupancy_log.db')

def load_data():
    """Reads the SQLite database and returns it as a Pandas DataFrame."""
    try:
        # timeout=10 prevents the 'database is locked' crash
        conn = sqlite3.connect(db_path, timeout=10) 
        
        # We use Pandas to execute the SQL query and format it into a data table
        df = pd.read_sql_query("SELECT * FROM traffic_events", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame() # Return empty table if it fails

# --- UI LAYOUT & REFRESH LOGIC ---
# We create an empty container so we can overwrite it with live data
placeholder = st.empty()

# This loop forces the dashboard to poll the database every 2 seconds
while True:
    df = load_data()
    
    with placeholder.container():
        if df.empty:
            st.warning("No data found in the database. Walk through the camera view to generate logs.")
        else:
            # Extract the most recent occupancy number
            current_occupancy = df.iloc[-1]['occupancy']
            total_in = len(df[df['event_type'] == 'IN'])
            total_out = len(df[df['event_type'] == 'OUT'])
            
            # --- TOP METRICS ROW ---
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Current Occupancy", value=current_occupancy)
            col2.metric(label="Total Daily Entries", value=total_in)
            col3.metric(label="Total Daily Exits", value=total_out)
            
            st.markdown("---")
            
            # --- DATA VISUALIZATION ---
            st.subheader("Occupancy Over Time")
            
            # Convert timestamp text to actual DateTime objects for graphing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Set the timestamp as the index so the chart knows what the X-axis is
            df.set_index('timestamp', inplace=True)
            
            # Draw the line chart tracking the 'occupancy' column
            st.line_chart(df['occupancy'])
            
            st.markdown("---")
            st.subheader("Raw Event Log")
            st.dataframe(df.tail(10)) # Show the 10 most recent crossings

    # Pause for 2 seconds before querying the database again
    time.sleep(2)