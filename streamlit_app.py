import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os

PROCESS_CSV = {
    'MAA – nouvelle substance active (procédure centralisée)': 'DayDataMaa.csv',
    'Variation Type IA / IB (procédure centralisée)': 'DayDataVariationTypeIAIB.csv',
    'Variation Type II (qualité ou nouvelle indication)': 'DayDataVariationType2.csv',
    'Extension d’AMM (nouvelle forme / dosage)': 'DayDataExtensionMaa.csv',
}

@st.cache_data
def load_data(process):
    path = PROCESS_CSV[process]
    if not os.path.exists(path):
        st.error(f"Fichier {path} introuvable.")
        st.stop()
    df = pd.read_csv(path)
    return df

def main():
    st.title('Suivi des Processus Réglementaires')
    process = st.sidebar.selectbox('Choisissez le type de process', list(PROCESS_CSV.keys()))
    st.header(f"Processus sélectionné : {process}")
    df = load_data(process)
    # Checklist : premier jalon de chaque groupe en gras
    st.subheader('Jalons à réaliser')
    checklist = {}
    for group in sorted(df['Concurrency_Group'].unique()):
        group_df = df[df['Concurrency_Group'] == group]
        first_step_no = group_df['Step_No'].min()
        first = pd.DataFrame(group_df[group_df['Step_No'] == first_step_no])
        if len(first) > 0:
            doc = str(first.iloc[0]['Document'])
            step_no = int(first.iloc[0]['Step_No'])
            st.markdown(f"**{doc}**")
            checked = st.checkbox(doc, key=f"step_{step_no}")
            checklist[step_no] = checked
        for _, row in group_df.iterrows():
            if row['Step_No'] == first_step_no:
                continue
            checked = st.checkbox(str(row['Document']), key=f"step_{row['Step_No']}")
            checklist[row['Step_No']] = checked

if __name__ == '__main__':
    main() 