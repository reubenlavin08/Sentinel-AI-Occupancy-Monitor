"""Sentinel multi-camera occupancy dashboard (Phase 5).

Per-camera + total occupancy, entries/exits, an occupancy-over-time chart, and a
recent-events log. Reads the WAL database without blocking the live writer.
Run:  streamlit run dashboard.py
"""
import streamlit as st
import sqlite3
import pandas as pd
import time
import os

st.set_page_config(page_title="Sentinel Occupancy", layout="wide")
st.title("Sentinel — Multi-Camera Occupancy")

script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "occupancy_log.db")


def load_data():
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        df = pd.read_sql_query("SELECT * FROM traffic_events", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


placeholder = st.empty()

while True:
    df = load_data()
    with placeholder.container():
        if df.empty:
            st.warning("No data yet — walk through a camera view to generate events.")
        else:
            if "camera" not in df.columns:
                df["camera"] = "unknown"
            df["camera"] = df["camera"].fillna("unknown")
            cams = sorted(df["camera"].unique())

            # current occupancy per camera = the latest occupancy value logged for it
            last = df.sort_values("id").groupby("camera").tail(1).set_index("camera")["occupancy"]
            total = int(last.sum())

            cols = st.columns(len(cams) + 1)
            cols[0].metric("TOTAL Occupancy", total)
            for i, cam in enumerate(cams):
                cols[i + 1].metric(f"{cam}", int(last.get(cam, 0)))

            st.markdown("---")

            left, right = st.columns(2)
            with left:
                st.subheader("Entries / Exits per camera")
                io = (df.groupby(["camera", "event_type"]).size()
                        .unstack(fill_value=0))
                st.dataframe(io, use_container_width=True)
            with right:
                st.subheader("Recent events")
                cols_show = [c for c in ["timestamp", "camera", "event_type", "occupancy"] if c in df.columns]
                st.dataframe(df.tail(15)[cols_show], use_container_width=True)

            st.markdown("---")
            st.subheader("Occupancy over time (per camera)")
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            pivot = (df.pivot_table(index="timestamp", columns="camera",
                                    values="occupancy", aggfunc="last")
                       .ffill())
            st.line_chart(pivot)

    time.sleep(2)
