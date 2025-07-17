import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Mapping "process label" -> CSV file
# ------------------------------------------------------------------
PROCESS_CSV = {
    "MAA – nouvelle substance active (procédure centralisée)": "DayDataMaa.csv",
    "Variation Type IA / IB (procédure centralisée)": "DayDataVariationTypeIAIB.csv",
    "Variation Type II (qualité ou nouvelle indication)": "DayDataVariationType2.csv",
    "Extension d’AMM (nouvelle forme / dosage)": "DayDataExtensionMaa.csv",
}


# ------------------------------------------------------------------
# Data loading (cached)
# ------------------------------------------------------------------
@st.cache_data
def load_data(process_label: str) -> pd.DataFrame:
    """Load the CSV for the selected process.

    * Ensures Prep_Days is numeric integer.
    * Leaves all other columns untouched.
    """
    path = PROCESS_CSV[process_label]
    if not os.path.exists(path):
        st.error(f"Fichier {path} introuvable.")
        st.stop()

    df = pd.read_csv(path)

    # Standardise column names we rely on (defensive: strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Ensure required columns exist
    required = {"Step_No", "Document", "Prep_Days", "Predecessor", "Concurrency_Group"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes dans {path} : {', '.join(sorted(missing))}.")
        st.stop()

    # Coerce types ---------------------------------------------------
    # Step_No: integer-like
    df["Step_No"] = pd.to_numeric(df["Step_No"], errors="coerce").astype("Int64")  # type: ignore

    # Concurrency_Group: integer-like; we'll keep Int64 but later cast to str for labels
    df["Concurrency_Group"] = pd.to_numeric(df["Concurrency_Group"], errors="coerce").astype("Int64")  # type: ignore

    # Prep_Days: strictly non-negative ints; NaN -> 0
    df["Prep_Days"] = pd.to_numeric(df["Prep_Days"], errors="coerce").fillna(0).clip(lower=0).astype(int)  # type: ignore

    return df


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
GROUP_COLOR_MAP = {
    "1": "#2ecc40",    # vert
    "2": "#0074d9",    # bleu
    "3": "#ff851b",    # orange
    "4": "#b10dc9",    # violet
    "5": "#ff4136",    # rouge
    "6": "#7fdbff",    # bleu clair
    "7": "#3d9970",    # vert foncé
    "8": "#f012be",    # rose
    "9": "#85144b",    # bordeaux
    "10": "#aaaaaa",   # gris
}


def compute_schedule(df: pd.DataFrame, j0_datetime: datetime) -> pd.DataFrame:
    """Return a dataframe with Start / Finish datetime columns computed by Concurrency_Group.

    Rules:
    - Project start = J0 - 365 days.
    - Group 1 starts at project_start.
    - Each subsequent group starts at the *max Finish* of the previous group.
    - Tasks within a group all start at the group start (i.e., fully parallel group model).
    - Finish = Start + Prep_Days (days).

    df is *not* modified in-place.
    """
    gantt_df = df.copy(deep=True)

    # Guarantee grouping column numeric & drop rows with missing group
    gantt_df = gantt_df[gantt_df["Concurrency_Group"].notna()].copy()

    # Work with plain ints for ordering
    group_order = sorted(gantt_df["Concurrency_Group"].dropna().astype(int).unique())  # type: ignore

    project_start = j0_datetime - timedelta(days=365)
    prev_group_max_finish = {}

    # We'll fill these columns with NaT initially
    gantt_df["Start"] = pd.NaT
    gantt_df["Finish"] = pd.NaT

    for idx, group in enumerate(group_order):
        if idx == 0:
            group_start = project_start
        else:
            # start when prior group completes
            prev_group = group_order[idx - 1]
            group_start = prev_group_max_finish[prev_group]

        mask = gantt_df["Concurrency_Group"].astype(int) == group
        # Assign start
        gantt_df.loc[mask, "Start"] = pd.to_datetime(group_start)
        # Finish = Start + duration
        gantt_df.loc[mask, "Finish"] = (
            gantt_df.loc[mask, "Start"]
            + pd.to_timedelta(gantt_df.loc[mask, "Prep_Days"], unit="D")
        )
        # Track max finish for sequencing the next group
        prev_group_max_finish[group] = gantt_df.loc[mask, "Finish"].max()

    # Final dtype normalisation
    gantt_df["Start"] = pd.to_datetime(gantt_df["Start"])
    gantt_df["Finish"] = pd.to_datetime(gantt_df["Finish"])

    # Sort for display
    gantt_df = gantt_df.sort_values(
        by=["Concurrency_Group", "Prep_Days", "Document"],
        ascending=[True, True, True],
    ).reset_index(drop=True)  # type: ignore

    return gantt_df


def eisenhower_metrics(gantt_df: pd.DataFrame) -> pd.DataFrame:
    """Add Slack, Urgence, Importance, Quadrant columns."""
    out = gantt_df.copy()

    # Slack: group max duration - task duration (parallel group assumption)
    out["Group_Max_Dur"] = out.groupby("Concurrency_Group")["Prep_Days"].transform("max")
    out["Slack_Days"] = out["Group_Max_Dur"] - out["Prep_Days"]

    # Urgence scale 0-10 (inverse slack)
    max_slack = out["Slack_Days"].max()
    if isinstance(max_slack, (pd.Series, np.ndarray)):
        max_slack = max_slack.item()
    if pd.isna(max_slack) or max_slack == 0:
        out["Urgence"] = 10.0
    else:
        out["Urgence"] = 10 * (1 - out["Slack_Days"] / max_slack)

    # Importance scale 0-10
    dur_max = out["Prep_Days"].max() or 1
    dur_scaled = out["Prep_Days"] / dur_max * 3  # 0-3 baseline
    out["On_Critical_Path"] = out["Slack_Days"] == 0
    out["Importance"] = np.where(
        out["On_Critical_Path"],
        np.minimum(7 + dur_scaled, 10),  # CP tasks boosted +7
        dur_scaled,
    )

    # Quadrant labels
    def quadrant(row):
        if row.Urgence >= 5 and row.Importance >= 5:
            return "Q1 Faire maintenant"
        if row.Urgence < 5 and row.Importance >= 5:
            return "Q2 Planifier"
        if row.Urgence >= 5 and row.Importance < 5:
            return "Q3 Déléguer"
        return "Q4 Éliminer"

    out["Quadrant"] = out.apply(quadrant, axis=1)
    return out


# ------------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Diagramme de Gantt – Suivi des Processus", layout="wide")
    st.title("Suivi des Processus Réglementaires & Diagramme de Gantt")

    # ------------------------------------------------------------------
    # Process selection + load
    # ------------------------------------------------------------------
    process = st.sidebar.selectbox("Choisissez le type de process", list(PROCESS_CSV.keys()))
    st.header(f"Processus sélectionné : {process}")
    df = load_data(process)

    # ------------------------------------------------------------------
    # Checklist (premier jalon de chaque groupe en gras)
    # ------------------------------------------------------------------
    st.subheader("Jalons à réaliser")
    checklist = {}
    for group in sorted(df["Concurrency_Group"].dropna().astype(int).unique()):
        group_mask = df["Concurrency_Group"].notna()
        group_df = df[group_mask & (df.loc[group_mask, "Concurrency_Group"].astype(int) == group)]
        first_step_no = group_df["Step_No"].min()
        first = group_df[group_df["Step_No"] == first_step_no]
        if not first.empty:  # type: ignore
            doc = str(first.iloc[0]["Document"])  # type: ignore
            step_no = int(first.iloc[0]["Step_No"])  # type: ignore
            st.markdown(f"**{doc}**")
            checked = st.checkbox(doc, key=f"step_{step_no}")
            checklist[step_no] = checked
        # Remaining steps in the group
        for _, row in group_df.iterrows():
            if row["Step_No"] == first_step_no:
                continue
            doc = str(row["Document"])
            checked = st.checkbox(doc, key=f"step_{int(row['Step_No'])}")
            checklist[int(row["Step_No"])] = checked

    # ------------------------------------------------------------------
    # J0 input (soumission)
    # ------------------------------------------------------------------
    st.subheader("Diagramme de Gantt")
    j0_date = st.date_input(
        "Sélectionnez la date J0 (date de soumission)",
        value=datetime.today(),
        help="La date J0 correspond à la date cible de soumission du dossier.",
    )
    if isinstance(j0_date, tuple):
        st.error("Veuillez sélectionner une seule date (pas une plage de dates) pour J0.")
        st.stop()
    if j0_date is None:
        st.error("Veuillez sélectionner une date valide pour J0.")
        st.stop()
    j0_datetime = datetime.combine(j0_date, datetime.min.time())

    # ------------------------------------------------------------------
    # Compute schedule
    # ------------------------------------------------------------------
    gantt_df = compute_schedule(df, j0_datetime)

    # ------------------------------------------------------------------
    # Interactive filters (Gantt)
    # ------------------------------------------------------------------
    with st.expander("Filtrer le diagramme de Gantt", expanded=False):
        st.markdown("### Filtres Gantt")
        available_groups_gantt = sorted(
            [int(g) for g in gantt_df["Concurrency_Group"].dropna().unique()],
            key=int,
        )
        selected_groups_gantt = st.multiselect(
            "Groupes (Concurrency_Group) à afficher (Gantt)",
            options=available_groups_gantt,
            default=available_groups_gantt,
            key="gantt_groups",
            help="Choisissez un ou plusieurs groupes ; les autres seront masqués.",
        )
        available_tasks_gantt = (
            gantt_df[gantt_df["Concurrency_Group"].isin(selected_groups_gantt)]["Document"].astype(str).tolist()
        )
        selected_tasks_gantt = st.multiselect(
            "Tâches (Document) à afficher – facultatif (Gantt)",
            options=available_tasks_gantt,
            default=available_tasks_gantt,
            key="gantt_tasks",
            help="Affinez encore la vue en sélectionnant des documents précis.",
        )

    gantt_df_filtered = gantt_df[
        gantt_df["Concurrency_Group"].isin(selected_groups_gantt)
        & gantt_df["Document"].isin(selected_tasks_gantt)
    ].copy()

    # ------------------------------------------------------------------
    # Optional detailed table
    # ------------------------------------------------------------------
    with st.expander("Afficher le tableau des tâches calculées"):
        st.dataframe(
            gantt_df_filtered[
                [
                    "Step_No",
                    "Document",
                    "Start",
                    "Finish",
                    "Prep_Days",
                    "Predecessor",
                    "Concurrency_Group",
                ]
            ]
        )

    # ------------------------------------------------------------------
    # Gantt chart
    # ------------------------------------------------------------------
    # Map group -> color (string keys for Plotly legend clarity)
    gantt_df_filtered["Concurrency_Group"] = gantt_df_filtered["Concurrency_Group"].astype(str)  # type: ignore
    color_map = {g: GROUP_COLOR_MAP.get(g, "#111111") for g in gantt_df_filtered["Concurrency_Group"].unique()}  # type: ignore

    fig = px.timeline(
        gantt_df_filtered,
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
        title="Diagramme de Gantt – Projet",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Document",
        margin=dict(l=0, r=20, t=40, b=0),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Eisenhower metrics + chart
    # ------------------------------------------------------------------
    st.subheader("Matrice d’Eisenhower – Priorisation des tâches")
    gantt_df_eisen = eisenhower_metrics(gantt_df)

    with st.expander("Filtrer la matrice d'Eisenhower", expanded=False):
        st.markdown("### Filtres Eisenhower")
        available_groups_eisen = sorted(
            [int(g) for g in gantt_df_eisen["Concurrency_Group"].dropna().unique()],
            key=int,
        )
        selected_groups_eisen = st.multiselect(
            "Groupes (Concurrency_Group) à afficher (Eisenhower)",
            options=available_groups_eisen,
            default=available_groups_eisen,
            key="eisen_groups",
            help="Choisissez un ou plusieurs groupes pour la matrice.",
        )
        available_tasks_eisen = (
            gantt_df_eisen[gantt_df_eisen["Concurrency_Group"].isin(selected_groups_eisen)]["Document"].astype(str).tolist()
        )
        selected_tasks_eisen = st.multiselect(
            "Tâches (Document) à afficher – facultatif (Eisenhower)",
            options=available_tasks_eisen,
            default=available_tasks_eisen,
            key="eisen_tasks",
            help="Affinez la matrice en sélectionnant des documents précis.",
        )

    gantt_df_eisen_filtered = gantt_df_eisen[
        gantt_df_eisen["Concurrency_Group"].isin(selected_groups_eisen)
        & gantt_df_eisen["Document"].isin(selected_tasks_eisen)
    ].copy()

    with st.expander("Afficher le tableau Eisenhower détaillé"):
        st.dataframe(
            gantt_df_eisen_filtered[
                [
                    "Step_No",
                    "Document",
                    "Prep_Days",
                    "Slack_Days",
                    "Urgence",
                    "Importance",
                    "Quadrant",
                    "Concurrency_Group",
                ]
            ]
        )

    fig_eisen = px.scatter(
        gantt_df_eisen_filtered,
        x="Urgence",
        y="Importance",
        text="Document",
        color="Quadrant",
        category_orders={
            "Quadrant": [
                "Q1 Faire maintenant",
                "Q2 Planifier",
                "Q3 Déléguer",
                "Q4 Éliminer",
            ]
        },
        hover_data={
            "Step_No": True,
            "Prep_Days": True,
            "Slack_Days": True,
            "Concurrency_Group": True,
            "Start": True,
            "Finish": True,
            "Quadrant": False,
        },
        labels={"Urgence": "Urgence (0–10)", "Importance": "Importance (0–10)"},
        title="Matrice d’Eisenhower",
        range_x=[-1, 11],
        range_y=[-1, 11],
    )
    fig_eisen.add_vline(x=5, line_dash="dash", line_color="gray")
    fig_eisen.add_hline(y=5, line_dash="dash", line_color="gray")
    fig_eisen.update_traces(textposition="top center")
    fig_eisen.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=40, b=0),
        showlegend=True,
    )
    st.plotly_chart(fig_eisen, use_container_width=True)


# ------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    main()
