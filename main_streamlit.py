"""Launching Hydraulic System monitoring app."""

# System imports.
import os
import numpy as np
import pandas as pd
import streamlit as st
import tkinter as tk
from tkinter import filedialog

# Package imports.
from hydraulic_systems.utils import load_model as hs_load_model

DATA_PATH = "data_subset"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data path not found: {DATA_PATH}")

VALVE_CONDITION = {
    100: "optimal switching behavior",
    90: "small lag",
    80: "severe lag",
    73: "close to total failure",
}



def select_file():
    root = tk.Tk()
    root.withdraw()
    # Make folder picker dialog appear on top of other windows
    root.wm_attributes("-topmost", 1)
    file = filedialog.askopenfile(master=root)
    if file:
        file_path = os.path.abspath(file.name)
    else:
        file_path = None
    root.destroy()
    return file_path


@st.cache_resource
def load_data():
    df_fs1 = pd.read_csv(
        os.path.join(DATA_PATH, "FS1.txt"), sep="\t", header=None
    )
    df_ps2 = pd.read_csv(
        os.path.join(DATA_PATH, "PS2.txt"), sep="\t", header=None
    )
    X_fs1 = df_fs1.to_numpy(np.float32)
    X_ps2 = df_ps2.to_numpy(np.float32)

    return X_fs1, X_ps2


@st.cache_resource
def load_model(path):
    """Wrapper for `load_model` to put the result in Streamlit cache."""
    model = hs_load_model(path)
    return model


def main():
    st.title("Valve condition prediction")

    st.header("Data and model")
    X_fs1, X_ps2 = load_data()

    st.write("Please select a fitted model.")
    file_select_button = st.button("Select file")
    model_path = st.session_state.get("model_path", None)
    if file_select_button:
        model_path = select_file()
        st.session_state.model_path = model_path
    if model_path:
        st.write("Selected file path:", model_path)

    if model_path:
        model = load_model(model_path)

        st.header("Prediction")
        st.write("Please select a cycle index")
        cycle_index = st.number_input(
            "Cycle index",
            min_value=0, max_value=len(X_fs1)-1
        )

        y = model.predict(
            X_fs1=X_fs1[cycle_index, :].reshape(1, -1),
            X_ps2=X_ps2[cycle_index, :].reshape(1, -1)
        )
        st.write(f"Valve condition: {y[0]}%")
        msg = VALVE_CONDITION[y[0]]
        if y[0] == 100:
            st.success(msg, icon="✅")
        elif y[0] < 80:
            st.error(msg, icon="🚨")
        else:
            st.warning(msg, icon="⚠️")


if __name__ == "__main__":
    main()