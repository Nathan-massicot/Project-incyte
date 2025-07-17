import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from io import StringIO
import datetime as dt

# -------------------------------------------------
# Données "hard-coded" (CSV) – à remplacer si besoin
# -------------------------------------------------
DATA = """Step_No,Document,Prep_Days,Predecessor,Concurrency_Group
1,Rapport développement pharmaceutique (extension),180,,1
2,CSR bioéquivalence,110,1,2
3,Module 3 (extension),90,1;2,2
4,Addendum QOS (extension),28,3,3
5,Addendum Clinical Overview (extension),20,2,3
6,SmPC/PL nouvelle forme,21,4;5,4
7,eAF Extension,7,6,4
8,Cover letter extension,2,7,5
9,Compilation eCTD (extension),14,8,6
10,Validation + Soumission (extension),3,9,7
"""

def load_data() -> pd.DataFrame:
    """Charge le CSV dans un DataFrame pandas et applique le typage adéquat."""
    df = pd.read_csv(StringIO(DATA))
    df["Prep_Days"] = df["Prep_Days"].astype(int)
    df["Concurrency_Group"] = df["Concurrency_Group"].astype(int)
    return df


# -----------------------------
# Mise en page Streamlit
# -----------------------------
st.set_page_config(page_title="Diagramme de Gantt – Extension", layout="wide")
st.title("Diagramme de Gantt – Planning d'extension")

planning_df = load_data()

# Sélection de la date J0 (soumission)
j0_date = st.date_input(
    "Sélectionnez la date J0 (date de soumission)",
    value=datetime.today(),
    help="La date J0 correspond à la date cible de soumission du dossier."
)

# Conversion explicite de j0_date en datetime
if isinstance(j0_date, tuple):
    st.error("Veuillez sélectionner une seule date (pas une plage de dates) pour J0.")
    st.stop()
if j0_date is None:
    st.error("Veuillez sélectionner une date valide pour J0.")
    st.stop()
j0_datetime = datetime.combine(j0_date, datetime.min.time())

# -----------------------------
# Calcul des dates Start / Finish
# -----------------------------
df = planning_df.copy()

group_order = sorted(df["Concurrency_Group"].unique())
prev_group_max_finish = {}

# Date de début projet = J0 moins 365 jours (modifiable)
project_start = j0_datetime - timedelta(days=365)

group_start_dates = {}
for i, group in enumerate(group_order):
    if i == 0:
        group_start = project_start
    else:
        group_start = prev_group_max_finish[group_order[i-1]]
    group_start_dates[group] = group_start
    mask = df["Concurrency_Group"] == group
    # Toutes les tâches du groupe démarrent à la même date
    df.loc[mask, "Start"] = group_start
    df.loc[mask, "Finish"] = df.loc[mask, "Start"] + df.loc[mask, "Prep_Days"].apply(lambda d: timedelta(days=int(d)))
    prev_group_max_finish[group] = df.loc[mask, "Finish"].max()

# Trier pour l'affichage : dans chaque groupe, la tâche la plus courte en haut, la plus longue en bas
df = df.sort_values(by=["Concurrency_Group", "Prep_Days", "Document"], ascending=[True, True, True]).reset_index(drop=True)

# Option : affichage tableau détaillé
a = st.expander("Afficher le tableau des tâches calculées")
with a:
    st.dataframe(
        df[[
            "Step_No",
            "Document",
            "Start",
            "Finish",
            "Prep_Days",
            "Predecessor",
            "Concurrency_Group",
        ]]
    )

# -----------------------------
# Palette de couleurs fixe par Concurrency_Group
# -----------------------------
# Couleurs fixes pour chaque groupe (ajuster si plus de groupes)
group_color_map = {
    "1": "#2ecc40",   # vert
    "2": "#0074d9",   # bleu
    "3": "#ff851b",   # orange
    "4": "#b10dc9",   # violet
    "5": "#ff4136",   # rouge
    "6": "#7fdbff",   # bleu clair
    "7": "#3d9970",   # vert foncé
    "8": "#f012be",   # rose
    "9": "#85144b",   # bordeaux
    "10": "#aaaaaa",  # gris
}
# Conversion en str pour forcer les couleurs catégorielles
# (à faire juste avant px.timeline)
df["Concurrency_Group"] = df["Concurrency_Group"].astype(str)
color_map = {g: group_color_map.get(g, "#111111") for g in df["Concurrency_Group"].unique()}

# -----------------------------
# Construction du diagramme de Gantt
# -----------------------------
# Trier pour l'affichage : dans chaque groupe, la tâche qui finit le plus tôt en haut
df = df.sort_values(by=["Concurrency_Group", "Finish", "Start"], ascending=[True, True, True]).reset_index(drop=True)

fig = px.timeline(
    df,
    x_start="Start",
    x_end="Finish",
    y="Document",
    color="Concurrency_Group",
    color_discrete_map=color_map,
    labels={"Concurrency_Group": "Etape simultanée"},
    hover_data={
        "Step_No": True,
        "Prep_Days": True,
        "Start": True,
        "Finish": True,
        "Concurrency_Group": True,
        "Predecessor": True,
    },
    title="Diagramme de Gantt – Projet d'extension",
)

# Placement de la première tâche en haut + suppression légende
fig.update_yaxes(autorange="reversed")
fig.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Document",
    margin=dict(l=0, r=20, t=40, b=0),
    showlegend=True,  # Affiche la légende Plotly
)

st.plotly_chart(fig, use_container_width=True)

