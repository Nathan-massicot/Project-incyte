import os
import re
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Mapping "process label" -> CSV file
# ------------------------------------------------------------------
PROCESS_CSV = {
    "MAA – new active substance (centralised)": "DayDataMaa.csv",
    "Variation Type IA / IB (centralised)": "DayDataVariationTypeIAIB.csv",
    "Variation Type II (quality or new indication)": "DayDataVariationType2.csv",
    "FDA (new form / strength)": "DayDataFDAMaa.csv",
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

    if "Effort_Days" not in df.columns:
        df["Effort_Days"] = df["Prep_Days"] * 0.8  # Par défaut : effort = 80% de la durée

    df.columns = [c.strip() for c in df.columns]

    required = {"Step_No", "Document", "Prep_Days", "Predecessor", "Concurrency_Group"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes dans {path} : {', '.join(sorted(missing))}.")
        st.stop()

    df["Step_No"] = pd.to_numeric(df["Step_No"], errors="coerce").astype("Int64")
    df["Concurrency_Group"] = pd.to_numeric(df["Concurrency_Group"], errors="coerce").astype("Int64")
    df["Prep_Days"] = pd.to_numeric(df["Prep_Days"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    return df

# ------------------------------------------------------------------
# Color mapping for Gantt chart groups
# ------------------------------------------------------------------
GROUP_COLOR_MAP = {
    "1": "#2ecc40",
    "2": "#0074d9",
    "3": "#ff851b",
    "4": "#b10dc9",
    "5": "#ff4136",
    "6": "#7fdbff",
    "7": "#3d9970",
    "8": "#f012be",
    "9": "#85144b",
    "10": "#aaaaaa",
}

# ------------------------------------------------------------------
# Compute Gantt schedule
# ------------------------------------------------------------------
def compute_schedule(df: pd.DataFrame, j0_datetime: datetime) -> pd.DataFrame:
    """
    Compute start and finish dates for tasks grouped by Concurrency_Group.
    Tasks in the same group start together. Each group starts after the previous one finishes.
    """
    gantt_df = df.copy(deep=True)
    gantt_df = gantt_df[gantt_df["Concurrency_Group"].notna()].copy()

    group_order = sorted(gantt_df["Concurrency_Group"].dropna().astype(int).unique())

    project_start = j0_datetime - timedelta(days=365)
    prev_group_max_finish = {}
    gantt_df["Start"] = pd.NaT
    gantt_df["Finish"] = pd.NaT

    for idx, group in enumerate(group_order):
        group_start = project_start if idx == 0 else prev_group_max_finish[group_order[idx - 1]]

        mask = gantt_df["Concurrency_Group"].astype(int) == group
        gantt_df.loc[mask, "Start"] = pd.to_datetime(group_start)
        gantt_df.loc[mask, "Finish"] = gantt_df.loc[mask, "Start"] + pd.to_timedelta(
            gantt_df.loc[mask, "Prep_Days"], unit="D"
        )
        prev_group_max_finish[group] = gantt_df.loc[mask, "Finish"].max()

    gantt_df["Start"] = pd.to_datetime(gantt_df["Start"])
    gantt_df["Finish"] = pd.to_datetime(gantt_df["Finish"])

    gantt_df = gantt_df.sort_values(
        by=["Concurrency_Group", "Prep_Days", "Document"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return gantt_df

# ------------------------------------------------------------------
# Compute weekly load (FTE)
# ------------------------------------------------------------------
def compute_weekly_load(df: pd.DataFrame, fte_available: float) -> pd.DataFrame:
    """
    Compute weekly load percentage based on Effort_Days and available FTE.
    Output: DataFrame with columns: Week, Total_Effort_Days, Load_%
    """
    expanded = []

    for _, row in df.iterrows():
        if pd.isna(row["Start"]) or pd.isna(row["Finish"]):
            continue

        try:
            start = row["Start"].date()
            end = row["Finish"].date()
        except Exception:
            continue

        days = max((end - start).days, 1)
        daily_effort = row["Effort_Days"] / days

        for i in range(days):
            current_day = start + timedelta(days=i)
            iso_year, iso_week, _ = current_day.isocalendar()
            expanded.append({
                "Week": f"{iso_year}-W{iso_week:02d}",
                "Date": current_day,
                "Effort_Days": daily_effort,
            })

    df_expanded = pd.DataFrame(expanded)
    weekly = (
        df_expanded.groupby("Week")
        .agg(Total_Effort_Days=("Effort_Days", "sum"))
        .reset_index()
    )
    weekly["Load_%"] = (weekly["Total_Effort_Days"] / (5 * fte_available)) * 100

    return weekly

# ------------------------------------------------------------------
# Compute urgency score
# ------------------------------------------------------------------
def compute_urgency(row):
    """Score d’urgence = 50 si critique + surcharge FTE > 100%"""
    score = 50 if row.get("Critical") else 0
    overload = max(0, row.get("Load_%", 0) - 100)
    return score + overload

# ------------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Gantt Chart – Process Tracking", layout="wide")
    st.title("Regulatory Process Tracking & Gantt Chart")

    with st.expander("Options", expanded=True):
        # 1. Dataset selection
        process = st.selectbox("Choose process type", list(PROCESS_CSV.keys()))
        df = load_data(process)

        # 2. Date J0 input
        planning_mode = st.radio(
            "Planning anchor point",
            ["Use J0 as submission date (classic)", "Use custom project start date"],
            index=0,
        )

        anchor_date = st.date_input(
            "Select anchor date (J0 or start)",
            value=datetime.today(),
        )

        if isinstance(anchor_date, tuple) or anchor_date is None:
            st.error("❌ Veuillez sélectionner une seule date valide.")
            st.stop()

        anchor_datetime = datetime.combine(anchor_date, datetime.min.time())
        reverse_from_j0 = planning_mode.startswith("Use J0")

        # 3. Compute schedule
        gantt_df = compute_schedule(
            df,
            anchor_datetime if reverse_from_j0 else anchor_datetime + timedelta(days=365)
        )

        # 4. Optional filtering
        if st.checkbox("Enable Gantt filters", value=False):
            safe_groups = [
                int(g) for g in gantt_df["Concurrency_Group"].dropna().unique()
                if str(g).isdigit()
            ]
            selected_groups = st.multiselect("Groups to display", options=safe_groups, default=safe_groups)

            available_tasks = gantt_df[
                gantt_df["Concurrency_Group"].isin(selected_groups)
            ]["Document"].astype(str).tolist()

            selected_tasks = st.multiselect("Tasks to display", options=available_tasks, default=available_tasks)

            gantt_df = gantt_df[
                gantt_df["Concurrency_Group"].isin(selected_groups)
                & gantt_df["Document"].isin(selected_tasks)
            ]

    # ---------- GANTT CHART ----------
    gantt_df["Concurrency_Group"] = gantt_df["Concurrency_Group"].astype(str)
    color_map = {g: GROUP_COLOR_MAP.get(g, "#111111") for g in gantt_df["Concurrency_Group"].unique()}

    fig = px.timeline(
        gantt_df,
        x_start="Start", x_end="Finish", y="Document",
        color="Concurrency_Group",
        color_discrete_map=color_map,
        title="📊 Gantt Chart – Project Timeline",
        hover_data={"Step_No": True, "Prep_Days": True, "Start": True, "Finish": True,
                    "Concurrency_Group": True, "Predecessor": True},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=600, margin=dict(l=0, r=20, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Project Duration ----------
    min_start, max_finish = gantt_df["Start"].min(), gantt_df["Finish"].max()
    if pd.isna(min_start) or pd.isna(max_finish):
        st.warning("⚠️ Impossible de calculer la durée du projet : dates manquantes.")
    else:
        duration = (max_finish - min_start).days
        st.markdown(
            f"**Project Duration:** {duration} days "
            f"(from `{min_start.date()}` to `{max_finish.date()}`)"
        )

    # ---------- FTE Load Analysis ----------
    with st.expander("📈 Load Analysis – FTE Simulation", expanded=False):
        fte = st.number_input("FTE available", min_value=0.1, value=1.0, step=0.1)
        weekly_load = compute_weekly_load(gantt_df, fte)

        # Ajout de Week_Label pour fusion
        gantt_df["Week_Label"] = gantt_df["Start"].dt.strftime("%G-W%V")
        gantt_df = gantt_df.merge(
            weekly_load[["Week", "Load_%"]],
            left_on="Week_Label", right_on="Week",
            how="left"
        )

        # Calcul slack & critical path
        gantt_df["Slack_Days"] = (
            gantt_df.groupby("Concurrency_Group")["Prep_Days"].transform("max")
            - gantt_df["Prep_Days"]
        )
        gantt_df["Critical"] = gantt_df["Slack_Days"] == 0
        gantt_df["Urgency_Score"] = gantt_df.apply(compute_urgency, axis=1)

        # Graph charge
        st.markdown("### Weekly Load (% of FTE capacity)")
        fig_load = px.line(
            weekly_load, x="Week", y="Load_%",
            title="Weekly Load (%) – Relative to Available FTE",
            markers=True
        )
        fig_load.add_hline(
            y=100, line_dash="dash", line_color="red",
            annotation_text="100% Capacity", annotation_position="top left"
        )
        fig_load.update_layout(height=400)
        st.plotly_chart(fig_load, use_container_width=True)
        
    # ---------- Critical Path ----------
    with st.expander("🔥 Show Critical Path Tasks", expanded=False):
        st.markdown("Les tâches critiques sont celles qui n'ont **aucune marge de retard (Slack = 0)**.")
        critical_tasks = gantt_df[gantt_df["Critical"]].copy()

        if critical_tasks.empty:
            st.info("✅ Aucune tâche critique détectée (toutes ont une marge suffisante).")
        else:
            display_cols = [
                "Step_No", "Document", "Prep_Days", "Effort_Days",
                "Start", "Finish", "Concurrency_Group", "Urgency_Score"
            ]
            rename_map = {
                "Step_No": "Step",
                "Prep_Days": "Duration (Days)",
                "Effort_Days": "Effort (FTE Days)",
                "Concurrency_Group": "Group"
            }
            df_display = critical_tasks[display_cols].rename(columns=rename_map)
            df_display["Start"] = df_display["Start"].dt.strftime("%Y-%m-%d")
            df_display["Finish"] = df_display["Finish"].dt.strftime("%Y-%m-%d")

            st.dataframe(df_display.reset_index(drop=True), use_container_width=True)
            
    # ---------- Safe export filename parts ----------
    safe_process = re.sub(r"[^\w]+", "_", process)
    anchor_label = "J0" if reverse_from_j0 else "Start"
    date_str = anchor_date.strftime("%Y-%m-%d")

    # ---------- Download buttons (Gantt, FTE, Critical Path) ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Download Gantt Schedule (CSV)",
            data=gantt_df.to_csv(index=False).encode("utf-8"),
            file_name=f"Gantt_{safe_process}_{anchor_label}_{date_str}.csv",
            mime="text/csv",
            key="download_gantt"
        )

    with col2:
        st.download_button(
            label="📥 Download Weekly FTE Load (CSV)",
            data=weekly_load.to_csv(index=False).encode("utf-8"),
            file_name=f"FTE_Weekly_Load_{safe_process}_{date_str}.csv",
            mime="text/csv",
            key="download_fte"
        )

    with col3:
        csv_critical = (
            gantt_df[gantt_df["Critical"]][
                ["Step_No", "Document", "Prep_Days", "Effort_Days",
                 "Start", "Finish", "Concurrency_Group", "Urgency_Score"]
            ]
            .rename(columns={
                "Step_No": "Step",
                "Prep_Days": "Duration (Days)",
                "Effort_Days": "Effort (FTE Days)",
                "Concurrency_Group": "Group"
            })
            .assign(
                Start=lambda df: pd.to_datetime(df["Start"], errors="coerce").dt.strftime("%Y-%m-%d"),
                Finish=lambda df: pd.to_datetime(df["Finish"], errors="coerce").dt.strftime("%Y-%m-%d"),
            )
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="📥 Download Critical Path (CSV)",
            data=csv_critical,
            file_name=f"Critical_Path_{safe_process}_{anchor_label}_{date_str}.csv",
            mime="text/csv",
            key="download_critical"
        )



if __name__ == "__main__":  # pragma: no cover
    main()
    