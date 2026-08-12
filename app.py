"""
DatoScope navigation entrypoint.

Run this file with Streamlit to get clean sidebar page names.
"""

from __future__ import annotations

import streamlit as st


navigation = st.navigation(
    [
        st.Page("pages/1_EDA.py", title="EDA"),
        st.Page("pages/2_Supervised_Modeling.py", title="Supervised Modeling"),
        st.Page("pages/3_Clustering.py", title="Clustering"),
        st.Page("pages/4_Comparison.py", title="Comparison"),
    ],
    position="sidebar",
)

navigation.run()
