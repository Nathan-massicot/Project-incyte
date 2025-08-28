import os
import re
from datetime import datetime, timedelta, date
import math
import csv
import hashlib

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure, Scatter

# ------------------------------------------------------------------
# Mapping "process label" -> CSV file (all in data/)
# ------------------------------------------------------------------
PROCESS_CSV = {
    "MAA Modules agregated": "data/DayDataMaav6_Modules_aggregated.csv",
    "MAA": "data/DayDataMaaV6.csv",
    "FDA NDA and BLA": "data/DayDataFDAMaa.csv",
    
}

TEAMS_MAP_PATH = "data/Regulatory_Departments_and_Teams.csv"

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data
def load_data(process_label: str) -> pd.DataFrame:
    path = PROCESS_CSV[process_label]
    if not os.path.exists(path):
        st.error(f"CSV not found: {path}")
        st.stop()

    # Robust CSV loading to tolerate embedded commas/quotes in Notes and occasional bad lines
    try:
        # Use Python engine up front to avoid C-engine 'Skipping line' ParserWarnings
        df = pd.read_csv(
            path,
            engine="python",
            sep=",",
            quotechar='"',
            doublequote=True,
            escapechar='\\',
            on_bad_lines="warn",     # warn but do not drop silently
            skipinitialspace=True,
            encoding_errors="replace"
        )
    except Exception as e1:
        try:
            # Last resort: be permissive about separators (handles stray semicolons)
            df = pd.read_csv(
                path,
                engine="python",
                sep=None,  # auto-detect
                quotechar='"',
                doublequote=True,
                escapechar='\\',
                on_bad_lines="skip",
                skipinitialspace=True,
                encoding_errors="replace"
            )
        except Exception as e2:
            st.error(f"Failed to read CSV {path}:\n1) {e1}\n2) {e2}")
            st.stop()
    df.columns = [c.strip() for c in df.columns]

    # --- Column compatibility aliasing (Task vs Document) ---
    if "Task" in df.columns and "Document" not in df.columns:
        df["Document"] = df["Task"]
    elif "Document" in df.columns and "Task" not in df.columns:
        df["Task"] = df["Document"]

    # Minimal columns (flexible: allow PERT or classic durations)
    required_min = {"Step_No", "Task"}
    miss_min = required_min - set(df.columns)
    if miss_min:
        st.error(f"Missing columns in {path}: {', '.join(sorted(miss_min))}")
        st.stop()

    # Ensure numeric types when present
    for col in ["Task_ID", "Step_No", "Prep_Days", "O_Days", "M_Days", "P_Days", "dispatch_date"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive Prep_Days if missing (fallback to M_Days or 0)
    if "Prep_Days" not in df.columns:
        if "M_Days" in df.columns:
            df["Prep_Days"] = df["M_Days"].fillna(0).astype(int)
        else:
            df["Prep_Days"] = 0
    else:
        df["Prep_Days"] = df["Prep_Days"].fillna(0).astype(int)

    # Normalize Use_PERT to boolean-like string (Y/N)
    if "Use_PERT" in df.columns:
        df["Use_PERT"] = df["Use_PERT"].astype(str).str.strip().str.upper()

    # Effort_Days
    if "Effort_Days" not in df.columns:
        df["Effort_Days"] = df["Prep_Days"] * 0.8
    else:
        df["Effort_Days"] = pd.to_numeric(df["Effort_Days"], errors="coerce").fillna(0)

    # Require dispatch_date for scheduling based on dispatch anchor
    if "dispatch_date" not in df.columns:
        st.error("Missing required column: dispatch_date")
        st.stop()
    df["dispatch_date"] = pd.to_numeric(df["dispatch_date"], errors="coerce").fillna(0).astype(int)

    # Filters columns
    if "Department" not in df.columns:
        df["Department"] = pd.NA
    if "Role" not in df.columns:
        df["Role"] = pd.NA

    # Concurrency group (kept for design/coloring & scheduling)
    if "Concurrency_Group" not in df.columns:
        df["Concurrency_Group"] = 1
    df["Concurrency_Group"] = pd.to_numeric(df["Concurrency_Group"], errors="coerce").fillna(1).astype(int)

    # Clean strings
    for c in ["Department", "Role", "Task", "Notes"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # --- Milestones column (flexible casing) ---
    # Normalize to a single column named "Milestones" with integer 0/1 values
    if "Milestones" not in df.columns and "milestones" in df.columns:
        df = df.rename(columns={"milestones": "Milestones"})
    if "Milestones" not in df.columns and "milestone" in df.columns:
        df = df.rename(columns={"milestone": "Milestones"})
    if "Milestones" not in df.columns:
        df["Milestones"] = 0
    df["Milestones"] = pd.to_numeric(df["Milestones"], errors="coerce").fillna(0).astype(int)

    # Stable order
    df = df.sort_values(by=["Step_No", "Task"], kind="stable").reset_index(drop=True)
    return df

@st.cache_data
def load_teams_map(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Department", "Role", "Team"])
    m = pd.read_csv(path)
    m.columns = [c.strip() for c in m.columns]
    if "Team / Task" in m.columns and "Team" not in m.columns:
        m = m.rename(columns={"Team / Task": "Team"})
    keep = [c for c in ["Department", "Role", "Team"] if c in m.columns]
    return m[keep].copy()

def attach_team(df: pd.DataFrame, teams_map: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"Department", "Role"}.issubset(out.columns) and {"Department", "Role"}.issubset(teams_map.columns):
        out = out.merge(teams_map, on=["Department", "Role"], how="left")
        if "Team" not in out.columns:
            out["Team"] = out["Role"]
        else:
            out["Team"] = out["Team"].fillna(out["Role"])
    else:
        out["Team"] = out["Role"]
    return out

# ------------------------------------------------------------------
# Scheduling helpers (PERT duration only, dispatch-based)
# ------------------------------------------------------------------
def _duration_days(row: pd.Series) -> int:
  """Compute task duration from PERT (O/M/P with Use_PERT) or fallback to Prep_Days.
  Returns an integer number of days >= 0.
  """
  # Prefer PERT if available and flagged
  if str(row.get("Use_PERT", "")).strip().upper() == "Y":
      o = row.get("O_Days")
      m = row.get("M_Days")
      p = row.get("P_Days")
      if pd.notna(o) and pd.notna(m) and pd.notna(p):
          try:
              est = (float(o) + 4.0 * float(m) + float(p)) / 6.0
              return max(0, int(math.ceil(est)))
          except Exception:
              pass
  # Fallback to M_Days then Prep_Days
  m = row.get("M_Days")
  if pd.notna(m):
      return max(0, int(math.ceil(float(m))))
  return max(0, int(row.get("Prep_Days", 0)))


# ------------------------------------------------------------------
# Display selection: pick exactly ONE row per task based on Effort_Days
# ------------------------------------------------------------------
def select_display_role_rows(df: pd.DataFrame, sel_roles: list) -> pd.DataFrame:
    """Return one row per task, choosing the display role dynamically.

    Rules:
    - Group by Task_ID if present (and not NA); else fallback to (Step_No, Document).
    - Within each task, rank roles by Effort_Days descending, then by original CSV order as tie-breaker.
    - Keep the first role that belongs to sel_roles (if sel_roles is empty, all roles are admissible).
    - If no admissible role remains for the task, exclude the task.
    Adds helper columns:
      - DisplayRoleEffort (float)
      - DisplayRoleFallback (bool), True if the chosen role isn't the #1 by Effort_Days.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    work = df.copy()
    # Preserve original CSV order (index is stable thanks to load_data sorting)
    work = work.reset_index().rename(columns={"index": "_csv_order"})

    # --- Build composite grouping key (ONE line per task) ---
    # Define a task uniquely by Step_No + Task (ignore Task_ID to avoid duplicates when Task_ID differs per role)
    work["_step"] = pd.to_numeric(work.get("Step_No"), errors="coerce").astype("Int64") if "Step_No" in work.columns else pd.Series([pd.NA] * len(work))
    work["_task"] = work["Task"].astype(str).str.strip() if "Task" in work.columns else pd.Series([""] * len(work))
    work["_grp_key"] = (
        work["_step"].astype("string").fillna("") + "||" +
        work["_task"].astype(str)
    )

    chosen_rows = []
    sel_roles_list = [str(r) for r in sel_roles] if sel_roles else []

    for _, sub in work.groupby("_grp_key", dropna=False):
        # Rank candidates within the task
        sub = sub.copy()
        sub["_eff"] = pd.to_numeric(sub.get("Effort_Days", 0), errors="coerce").fillna(0.0)
        sub = sub.sort_values(by=["_eff", "_csv_order"], ascending=[False, True])
        # Determine preferred order
        preferred = sub
        # Apply role admissibility if roles filter is active
        if sel_roles_list:
            cand = preferred[preferred["Role"].astype(str).isin(sel_roles_list)]
        else:
            cand = preferred
        if cand.empty:
            # No admissible role for this task -> drop the task
            continue
        chosen = cand.iloc[0]
        # Fallback flag: compare to absolute best (first of preferred)
        is_fallback = bool(chosen.name != preferred.index[0])
        chosen = chosen.copy()
        chosen["DisplayRoleEffort"] = float(chosen["_eff"]) if pd.notna(chosen["_eff"]) else 0.0
        chosen["DisplayRoleFallback"] = is_fallback
        chosen_rows.append(chosen)

    if not chosen_rows:
        return pd.DataFrame(columns=df.columns)

    out = pd.DataFrame(chosen_rows).copy()
    # Clean helper columns
    for c in ["_csv_order", "_grp_key", "_eff", "_step", "_task"]:
        if c in out.columns:
            del out[c]
    return out

# ------------------------------------------------------------------
# Schedule computation
# ------------------------------------------------------------------
def compute_schedule(df: pd.DataFrame, anchor_datetime: datetime, reverse_to_j0: bool = True) -> pd.DataFrame:
    """
    Schedule tasks using a *dispatch-based* approach (no predecessors).

    Columns used:
      - dispatch_date: integer number of days **before dispatch/J0** when the task is expected to **start**.
      - O_Days, M_Days, P_Days, Use_PERT (optional): used to compute duration via _duration_days.
      - Prep_Days as fallback duration if PERT not usable.

    Modes:
      - reverse_to_j0=True  → anchor_datetime is the **dispatch/J0 date**.
        Start = J0 - dispatch_date; Finish = Start + duration.

      - reverse_to_j0=False → anchor_datetime is a **custom project start**.
        We build a relative schedule where the earliest Start aligns to anchor_datetime.
        Concretely, Start is derived from dispatch_date (relative), then all tasks are shifted so that
        min(Start) == anchor_datetime.
    """
    if df is None or df.empty:
        out = df.copy() if df is not None else pd.DataFrame()
        if out is not None:
            out["Start"] = pd.NaT
            out["Finish"] = pd.NaT
        return out

    work = df.copy(deep=True)

    # Ensure dispatch_date is present & numeric
    if "dispatch_date" not in work.columns:
        st.error("Missing required column: dispatch_date")
        st.stop()
    work["dispatch_date"] = pd.to_numeric(work["dispatch_date"], errors="coerce").fillna(0).astype(int)

    # Compute duration per *task row* (role granularity retained)
    durations = work.apply(_duration_days, axis=1).astype(int)

    # Provisional Start/Finish according to the selected anchor mode
    if reverse_to_j0:
        # Anchor is J0 (dispatch/submission date)
        # Start = J0 - dispatch_date; Finish = Start + duration
        work["Start"] = work["dispatch_date"].apply(lambda d: anchor_datetime - timedelta(days=int(max(0, d))))
        work["Finish"] = work["Start"] + durations.apply(lambda d: timedelta(days=int(max(0, d))))
    else:
        # Custom project start: build a relative timeline, then shift so min(Start) == anchor_datetime
        # Use a relative base where the task with the largest dispatch_date starts earliest.
        max_disp = int(work["dispatch_date"].max()) if not work["dispatch_date"].empty else 0
        # Relative start in days from an arbitrary origin (0 at the earliest start)
        rel_start_days = max_disp - work["dispatch_date"].astype(int)
        base_origin = datetime(2000, 1, 1)
        work["Start"] = rel_start_days.apply(lambda d: base_origin + timedelta(days=int(max(0, d))))
        work["Finish"] = work["Start"] + durations.apply(lambda d: timedelta(days=int(max(0, d))))

        # Shift so that the earliest Start equals the chosen custom project start date
        min_start = pd.to_datetime(work["Start"], errors="coerce").min()
        if pd.notna(min_start):
            delta = anchor_datetime - min_start
            work["Start"] = pd.to_datetime(work["Start"]) + delta
            work["Finish"] = pd.to_datetime(work["Finish"]) + delta

    return work

# ------------------------------------------------------------------
# Weekly FTE load (line chart)
# ------------------------------------------------------------------
def compute_weekly_load(df: pd.DataFrame, fte_available: float) -> pd.DataFrame:
    expanded = []
    for _, row in df.iterrows():
        if pd.isna(row.get("Start")) or pd.isna(row.get("Finish")):
            continue
        s = pd.to_datetime(row["Start"]).date()
        e = pd.to_datetime(row["Finish"]).date()
        days = max((e - s).days, 1)
        daily_effort = float(row.get("Effort_Days", 0.0)) / days if days > 0 else 0.0
        for i in range(days):
            d = s + timedelta(days=i)
            y, w, _ = d.isocalendar()
            expanded.append({"Week": f"{y}-W{w:02d}", "Effort_Days": daily_effort})

    if not expanded:
        return pd.DataFrame(columns=["Week", "Total_Effort_Days", "Load_%"])

    wk = pd.DataFrame(expanded).groupby("Week", as_index=False)["Effort_Days"].sum()
    wk = wk.rename(columns={"Effort_Days": "Total_Effort_Days"})
    wk["Load_%"] = (wk["Total_Effort_Days"] / (5 * max(fte_available, 0.0001))) * 100
    return wk

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Process Organization Methodology - POM", layout="wide")
    st.title("Process Organization Methodology - POM")

    # ----- Options -----
    with st.expander("Options", expanded=True):
        process = st.selectbox("Choose process type", list(PROCESS_CSV.keys()), index=0)
        planning_mode = st.radio(
            "Planning anchor point",
            ["Use J0 as submission date (classic)", "Use custom project start date"],
            index=0,
        )
        anchor_date = st.date_input("Select anchor date (J0 or start)", value=date.today())
        if not isinstance(anchor_date, date):
            st.error("❌ Please select a valid date.")
            st.stop()
        anchor_dt = datetime.combine(anchor_date, datetime.min.time())
        reverse_to_j0 = planning_mode.startswith("Use J0")
        row_px = 15

    # Load + schedule
    base_df = load_data(process)
    gantt_df = compute_schedule(base_df, anchor_dt, reverse_to_j0)

    # ----- Gantt Filters (hidden by default) -----
    with st.expander("🔎 Gantt Filters", expanded=False):
        # Roles (multiselect)
        if "Role" in gantt_df.columns:
            role_opts = (
                sorted(gantt_df["Role"].dropna().astype(str).unique().tolist())
                if not gantt_df["Role"].dropna().empty else []
            )
            sel_roles = st.multiselect(
                "Roles",
                options=role_opts,
                default=role_opts
            ) if role_opts else []
        else:
            sel_roles = []

        # Task dropdown (single select) — independent from roles
        st.markdown("**Task filter**")
        task_base = gantt_df

        if "Task" in task_base.columns:
            task_values = (
                sorted(task_base["Task"].dropna().astype(str).unique().tolist())
                if not task_base["Task"].dropna().empty else []
            )
        else:
            task_values = []

        task_options = ["All tasks"] + task_values if task_values else ["All tasks"]
        selected_task = st.selectbox("Task to display", options=task_options, index=0)

    # Apply Gantt filters
    filt = gantt_df.copy()


    # Task filter (single selection)
    if selected_task != "All tasks":
        filt = filt[filt["Task"].astype(str) == str(selected_task)]

    # --- Build per-task display view based on selected roles ---
    filt_for_gantt = select_display_role_rows(filt, sel_roles)
    if filt_for_gantt.empty:
        st.warning("⚠️ No tasks match the current role selection.")
        return
    # --- Robust de-duplication by Task Key (Step_No + Task only) ---
    tmp = filt_for_gantt.copy()
    tmp["_step"] = pd.to_numeric(tmp.get("Step_No"), errors="coerce").astype("Int64") if "Step_No" in tmp.columns else pd.Series([pd.NA] * len(tmp))
    tmp["_taskcol"] = tmp["Task"].astype(str).str.strip() if "Task" in tmp.columns else pd.Series([""] * len(tmp))

    tmp["_task_key"] = (
        tmp["_step"].astype("string").fillna("") + "||" +
        tmp["_taskcol"].astype(str)
    )

    filt_for_gantt = (
        tmp.drop_duplicates(subset=["_task_key"], keep="first")
           .drop(columns=["_task_key", "_step", "_taskcol"], errors="ignore")
    )

    # Add TaskLabel: ⓘ before the task name if Notes is non-empty
    filt_for_gantt["TaskLabel"] = filt_for_gantt.apply(
        lambda r: f"ⓘ {r['Task']}" if str(r.get("Notes", "")).strip() != "" else r["Task"],
        axis=1
    )

    # ----- Gantt: one bar per task + chronological order -----
    gantt_plot = filt_for_gantt.copy()

    # Ensure datetimes and sort chronologically
    gantt_plot["Start"] = pd.to_datetime(gantt_plot["Start"], errors="coerce")
    gantt_plot["Finish"] = pd.to_datetime(gantt_plot["Finish"], errors="coerce")
    # Compute duration from scheduled dates for sorting (longer first when same Start)
    if "Start" in gantt_plot.columns and "Finish" in gantt_plot.columns:
        gantt_plot["Duration_Days"] = (gantt_plot["Finish"] - gantt_plot["Start"]).dt.days
        gantt_plot["Duration_Days"] = gantt_plot["Duration_Days"].fillna(0).clip(lower=0)
    else:
        gantt_plot["Duration_Days"] = 0
    # Sort: Start (asc), then longer tasks first when same Start
    desired_order = ["Start", "Duration_Days", "Finish", "Concurrency_Group", "Step_No", "Task"]
    sort_cols = [c for c in desired_order if c in gantt_plot.columns]
    ascending_map = {"Start": True, "Duration_Days": False, "Finish": True, "Concurrency_Group": True, "Step_No": True, "Task": True}
    ascending_list = [ascending_map[c] for c in sort_cols]
    gantt_plot = gantt_plot.sort_values(by=sort_cols, ascending=ascending_list, na_position="last").reset_index(drop=True)

    # Color by concurrency group (keep original look of thick lines)
    color_map = {
        "1": "#2ecc40", "2": "#0074d9", "3": "#ff851b", "4": "#b10dc9", "5": "#ff4136",
        "6": "#7fdbff", "7": "#3d9970", "8": "#f012be", "9": "#85144b", "10": "#aaaaaa",
    }
    # Couleurs fixes pour les équipes de Regulatory Affairs
    RA_COLORS = {
        "Regulatory Operations": "#d62728",              # rouge
        "Management": "#1f77b4",                         # bleu clair
        "Other function": "#8400ff",                     # violet
        "Regulatory CMC": "#ff7f0e",                     # orange
        "Global Regulatory Lead": "#2ca02c",             # vert
        "Regulatory Strategist (Global/EU)": "#8c564b",  # marron
        "EMA": "#bcbd22",                                # jaune
        "Labeling": "#C59F9F",                           # bleu foncé
    }

    # Deterministic base palette (fixed order). Used for unknown roles.
    BASE_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    def role_to_color(role_name: str) -> str:
        """Return a stable color for a given role across reloads.
        Priority: fixed RA_COLORS -> deterministic hash index into BASE_PALETTE.
        Uses MD5 of the role string (not Python's hash) to be stable across sessions.
        """
        r = (role_name or "").strip()
        if r in RA_COLORS:
            return RA_COLORS[r]
        # Deterministic index from md5 hex
        h = hashlib.md5(r.encode("utf-8")).hexdigest()
        idx = int(h, 16) % len(BASE_PALETTE)
        return BASE_PALETTE[idx]

    fig = Figure()

    # Colors are resolved deterministically per role via role_to_color();
    # this prevents color changes between Streamlit reloads.

    # Track which roles already appear in the legend to avoid duplicates
    seen_roles = set()

    # Milestone legend flag (to avoid duplicate legend entries)
    milestone_legend_added = False

    for i, (_, row) in enumerate(gantt_plot.iterrows()):
        dept = str(row.get("Department", "")).strip()
        role = str(row.get("Role", "")).strip()
        group_key = str(row.get("Concurrency_Group", ""))

        # One fixed color per Role (no department filtering)
        chosen_color = role_to_color(role) if role else "#888888"

        # Legend: one entry per Role (first occurrence only)
        legend_name = role if role else row["Task"]
        legend_group = role if role else None
        show_legend = False
        if role and role not in seen_roles:
            show_legend = True
            seen_roles.add(role)

        fig.add_trace(Scatter(
            x=[row["Start"], row["Finish"]],
            y=[i, i],
            mode="lines",
            line=dict(color=chosen_color, width=10),
            hovertemplate=(
                f"Display role auto = {role} (Effort_Days = {float(row.get('DisplayRoleEffort', row.get('Effort_Days', 0))):.2f}"
                f"{' – fallback' if bool(row.get('DisplayRoleFallback', False)) else ''})<br>"
                f"<b>{row['Task']}</b><br>"
                f"Start: {pd.to_datetime(row['Start']).date()}<br>"
                f"Finish: {pd.to_datetime(row['Finish']).date()}<br>"
                f"Duration: {int(row['Prep_Days']) if pd.notna(row['Prep_Days']) else 0} days<br>"
                f"Group: {row.get('Concurrency_Group','')}<br>"
                f"Dept: {row.get('Department','')}, Role: {row.get('Role','')}<extra></extra>"
            ),
            name=legend_name,
            showlegend=show_legend,
            legendgroup=legend_group
        ))

        # If this task is marked as a Milestone, add a diamond marker at the start
        is_milestone = False
        if "Milestones" in row.index:
            try:
                is_milestone = int(row["Milestones"]) == 1
            except Exception:
                is_milestone = False
        if is_milestone:
            fig.add_trace(Scatter(
                x=[row["Start"]],
                y=[i],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=14,
                    line=dict(width=1),
                    color=chosen_color
                ),
                hovertemplate=(
                    f"<b>{row['Task']}</b><br>Milestone at start<br>Start: {pd.to_datetime(row['Start']).date()}<extra></extra>"
                ),
                name="Milestone",
                showlegend=(not milestone_legend_added),
                legendgroup="Milestone"
            ))
            milestone_legend_added = True


    fig.update_yaxes(
        tickvals=list(range(len(gantt_plot))),
        ticktext=gantt_plot["TaskLabel"],
        autorange="reversed",
        tickfont=dict(size=9),
    )
    # Dynamic height & margins with padding for small selections
    num_rows = max(1, len(gantt_plot))
    base_height = 260        # ensures comfortable space for 1–3 tasks
    per_row_height = max(18, int(row_px))
    fig_height = max(base_height, per_row_height * num_rows + 80)

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Task",
        margin=dict(l=60, r=30, t=30, b=60),
        height=int(fig_height),
        showlegend=True,
    )

    # Add extra vertical padding on y when only one task is shown
    if num_rows == 1:
        fig.update_yaxes(range=[-0.8, 0.8])
    st.plotly_chart(fig, use_container_width=True)

    # Project duration summary
    min_start = pd.to_datetime(filt_for_gantt["Start"]).min()
    max_finish = pd.to_datetime(filt_for_gantt["Finish"]).max()
    if pd.notna(min_start) and pd.notna(max_finish):
        st.markdown(
            f"**Project Duration:** {(max_finish - min_start).days} days "
            f"(from `{min_start.date()}` to `{max_finish.date()}`)"
        )

    # ----- FTE (line) with Teams multiselect -----
    with st.expander("📈 Load Analysis – FTE Simulation", expanded=False):
        teams_map = load_teams_map(TEAMS_MAP_PATH)
        fte_src = attach_team(filt, teams_map)

        team_opts = sorted(fte_src["Team"].dropna().unique().tolist()) if "Team" in fte_src.columns else []
        sel_teams = st.multiselect("Teams for FTE graphs", options=team_opts, default=team_opts) if team_opts else []
        if sel_teams:
            fte_src = fte_src[fte_src["Team"].isin(sel_teams)]

        fte_available = st.number_input("FTE available", min_value=0.1, value=1.0, step=0.1)
        weekly = compute_weekly_load(fte_src, fte_available)

        st.markdown("### Weekly Load (% of FTE capacity)")
        fig_load = px.line(weekly, x="Week", y="Load_%",
                           title="Weekly Load (%) – Relative to Available FTE", markers=True)
        fig_load.add_hline(y=100, line_dash="dash", line_color="red",
                           annotation_text="100% Capacity", annotation_position="top left")
        fig_load.update_layout(height=400)
        st.plotly_chart(fig_load, use_container_width=True)

    # ----- FTE Calculator (dropdown, annualized) -----
    with st.expander("🧮 FTE Calculator (by Role/Task)", expanded=False):
        st.markdown(
            "This calculator annualizes the **Effort_Days** into FTE using:\n"
            r"$\mathrm{FTE}_{task} = \dfrac{\mathrm{Effort\\Days}}{\mathrm{Working\\Days\\ per\\FTE}}$"
        )

        # Working days per FTE (default 220)
        working_days_per_fte = st.number_input(
            "Working days per FTE (per year)",
            min_value=1, max_value=365, value=220, step=1
        )

        if "Role" not in filt.columns:
            st.warning("No 'Role' column found in the current dataset. Cannot compute FTE by role.")
        else:
            # Prepare task-level view (each row = role-task) restricted by current filters
            task_view_cols = [c for c in ["Role", "Department", "Task", "Effort_Days", "Start", "Finish"] if c in filt.columns]
            task_view = filt[task_view_cols].copy()

            # Ensure numerics
            task_view["Effort_Days"] = pd.to_numeric(task_view.get("Effort_Days", 0), errors="coerce").fillna(0.0)

            # Step 2: FTE per task (annualized)
            task_view["FTE_Task"] = task_view["Effort_Days"] / float(working_days_per_fte)

            # Optional cost: merge a rate table if present
            rate_col_name = "Rate_per_FTE_Day"
            rate_path = os.path.join("data", "Rate_per_FTE_Day.csv")
            rates_df = None
            if os.path.exists(rate_path):
                try:
                    rates_df = pd.read_csv(rate_path)
                    # normalize columns
                    rates_df.columns = [c.strip() for c in rates_df.columns]
                    if "Role" in rates_df.columns and rate_col_name in rates_df.columns:
                        task_view = task_view.merge(
                            rates_df[["Role", rate_col_name]], on="Role", how="left"
                        )
                        task_view["Cost"] = task_view["Effort_Days"] * task_view[rate_col_name].fillna(0.0)
                except Exception as e:
                    st.warning(f"Rate table found but could not be read: {e}")

            # Step 3: Verify capacity by role
            cap_path = os.path.join("data", "HR_Capacity.csv")
            cap_df = None
            if os.path.exists(cap_path):
                try:
                    cap_df = pd.read_csv(cap_path)
                    cap_df.columns = [c.strip() for c in cap_df.columns]
                    # We expect columns: Role, Available_FTE
                    if not {"Role", "Available_FTE"}.issubset(cap_df.columns):
                        st.warning("`HR_Capacity.csv` does not contain columns: Role, Available_FTE. Skipping capacity merge.")
                        cap_df = None
                except Exception as e:
                    st.warning(f"Capacity file found but could not be read: {e}")
                    cap_df = None
            else:
                cap_df = None

            # Aggregate per role
            agg_specs = {
                "Task": "nunique",
                "Effort_Days": "sum",
                "FTE_Task": "sum",
            }
            if "Cost" in task_view.columns:
                agg_specs["Cost"] = "sum"

            role_agg = (
                task_view.groupby("Role", as_index=False)
                .agg(**{
                    "Tasks": ("Task", "nunique"),
                    "Total_Effort_Days": ("Effort_Days", "sum"),
                    "Required_FTE": ("FTE_Task", "sum"),
                    **({"Estimated_Cost": ("Cost", "sum")} if "Cost" in task_view.columns else {})
                })
            )

            # Merge capacity if available
            if cap_df is not None:
                role_agg = role_agg.merge(cap_df[["Role", "Available_FTE"]], on="Role", how="left")
                role_agg["Available_FTE"] = pd.to_numeric(role_agg["Available_FTE"], errors="coerce")
                role_agg["Utilization_%"] = (role_agg["Required_FTE"] / role_agg["Available_FTE"]) * 100.0
                role_agg["Utilization_%"] = role_agg["Utilization_%"].replace([float("inf"), -float("inf")], pd.NA)
                role_agg["Shortfall_FTE"] = (role_agg["Required_FTE"] - role_agg["Available_FTE"]).clip(lower=0)
            else:
                role_agg["Available_FTE"] = pd.NA
                role_agg["Utilization_%"] = pd.NA
                role_agg["Shortfall_FTE"] = pd.NA

            # Display results
            st.markdown("### 📋 FTE per Task (annualized)")
            show_task_cols = [c for c in ["Role", "Department", "Task", "Effort_Days", "FTE_Task", "Start", "Finish"] if c in task_view.columns]
            if "Cost" in task_view.columns:
                show_task_cols += ["Cost"]
            st.dataframe(task_view[show_task_cols].reset_index(drop=True), use_container_width=True)

            st.markdown("### 🧑‍💼 FTE by Role (capacity check)")
            display_cols = ["Role", "Tasks", "Total_Effort_Days", "Required_FTE", "Available_FTE", "Utilization_%", "Shortfall_FTE"]
            if "Estimated_Cost" in role_agg.columns:
                display_cols.append("Estimated_Cost")

            # Sort by highest utilization / shortfall
            sort_cols = []
            if "Utilization_% " in role_agg.columns:
                sort_cols = ["Utilization_%"]
            elif "Required_FTE" in role_agg.columns:
                sort_cols = ["Required_FTE"]

            st.dataframe(role_agg[display_cols].sort_values(by=sort_cols or ["Required_FTE"], ascending=False).reset_index(drop=True),
                         use_container_width=True)

            # Downloads
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_process = re.sub(r"[^A-Za-z0-9]+", "_", process).strip("_")

            col_task, col_role = st.columns(2)

            with col_task:
                st.download_button(
                    "📥 Download FTE per Task (CSV)",
                    data=task_view[show_task_cols].to_csv(index=False),
                    file_name=f"FTE_per_Task_{safe_process}_{date_str}.csv",
                    mime="text/csv",
                    key="dl_fte_task"
                )

            with col_role:
                st.download_button(
                    "📥 Download FTE by Role (CSV)",
                    data=role_agg[display_cols].to_csv(index=False),
                    file_name=f"FTE_by_Role_{safe_process}_{date_str}.csv",
                    mime="text/csv",
                    key="dl_fte_role"
                )

    # ----- Critical Path (task-level, deduplicated, chronological) -----
    with st.expander("🔥 Show Critical Path Tasks", expanded=False):
        # Build task-level view to avoid duplicates
        grp_keys = ["Concurrency_Group"]
        if "Step_No" in filt.columns: grp_keys.append("Step_No")
        if "Task" in filt.columns: grp_keys.append("Task")

        if not {"Concurrency_Group", "Task"}.issubset(filt.columns):
            st.info("Columns required for critical path not found.")
        else:
            task_level = (
                filt.groupby(grp_keys, as_index=False)
                    .agg(
                        Prep_Days   = ("Prep_Days", "max"),   # task duration
                        Effort_Days = ("Effort_Days", "sum"), # total effort across roles
                        Start       = ("Start", "min"),       # earliest start
                        Finish      = ("Finish", "max")       # latest finish
                    )
            )

            # Slack per group from task-level durations
            task_level["Slack_Days"] = (
                task_level.groupby("Concurrency_Group")["Prep_Days"].transform("max")
                - task_level["Prep_Days"]
            )
            task_level["Critical"] = task_level["Slack_Days"] == 0

            # Build output DataFrame as before, but keep all columns for filtering
            out = task_level.rename(columns={
                "Step_No": "Step",
                "Prep_Days": "Duration (Days)",
                "Effort_Days": "Effort (FTE Days)",
                "Concurrency_Group": "Group",
            }).copy()

            # Ensure datetimes and sort chronologically
            if "Start" in out.columns:
                out["Start"] = pd.to_datetime(out["Start"], errors="coerce")
            if "Finish" in out.columns:
                out["Finish"] = pd.to_datetime(out["Finish"], errors="coerce")

            if out.empty:
                st.info("✅ Aucune tâche critique détectée (toutes ont une marge suffisante).")
            else:
                # Keep only critical tasks
                crit_only = out[out["Critical"] == True].copy()
                if crit_only.empty:
                    st.info("✅ Aucune tâche critique détectée (toutes ont une marge suffisante).")
                else:
                    crit_only = crit_only.sort_values(by=["Start", "Step", "Task"], ascending=[True, True, True])
                    # Drop the boolean column from display since everything is critical
                    display_cols = [c for c in ["Task_ID", "Step", "Task", "Duration (Days)", "Start", "Finish", "Slack_Days"] if c in crit_only.columns]
                    # Pretty date display
                    if "Start" in crit_only.columns:
                        crit_only["Start"] = crit_only["Start"].dt.strftime("%Y-%m-%d")
                    if "Finish" in crit_only.columns:
                        crit_only["Finish"] = crit_only["Finish"].dt.strftime("%Y-%m-%d")
                    st.dataframe(crit_only[display_cols].reset_index(drop=True), use_container_width=True)

                    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_process = re.sub(r"[^A-Za-z0-9]+", "_", process).strip("_")
                    st.download_button(
                        label="📥 Download Critical Path (CSV)",
                        data=crit_only[display_cols].to_csv(index=False),
                        file_name=f"Critical_Path_{safe_process}_{date_str}.csv",
                        mime="text/csv",
                        key="download_critical_tasks"
                    )

if __name__ == "__main__":
    main()
    