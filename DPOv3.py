import os
import re
from datetime import datetime, timedelta, date
import math

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure, Scatter

# ------------------------------------------------------------------
# Mapping "process label" -> CSV file (all in data/)
# ------------------------------------------------------------------
PROCESS_CSV = {
    "MAA": "data/DayDataMaa_enriched_pert.csv",
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

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Minimal columns (flexible: allow PERT or classic durations)
    required_min = {"Step_No", "Document"}
    miss_min = required_min - set(df.columns)
    if miss_min:
        st.error(f"Missing columns in {path}: {', '.join(sorted(miss_min))}")
        st.stop()

    # Ensure numeric types when present
    for col in ["Task_ID", "Step_No", "Prep_Days", "O_Days", "M_Days", "P_Days"]:
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
    for c in ["Department", "Role", "Document"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Stable order
    df = df.sort_values(by=["Step_No", "Document"], kind="stable").reset_index(drop=True)
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
# Advanced scheduling helpers (PERT + PDM dependencies)
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

def _parse_predecessors(pred_str: str) -> list:
  """Parse a Predecessors string like '3FS+0d;2SS-2d' -> [(3,'FS',0), (2,'SS',-2)]."""
  if not isinstance(pred_str, str) or not pred_str.strip():
      return []
  out = []
  parts = [p.strip() for p in pred_str.split(';') if p.strip()]
  for p in parts:
      # Extract leading ID
      m = re.match(r"^(\d+)([A-Za-z]{2})?([+-]\d+)?d?$", p)
      if not m:
          # Try more flexible parsing: ID + Type + +lag/-lag + optional 'd'
          m = re.match(r"^(\d+)\s*([FS|SS|FF|SF]{2})?\s*([+-]\s*\d+)\s*d?$", p, re.IGNORECASE)
      if m:
          pid = int(m.group(1))
          typ = (m.group(2) or "FS").upper()
          lag = m.group(3)
          lag_days = int(lag.replace(" ", "")) if lag else 0
          out.append((pid, typ, lag_days))
      else:
          # Fallback: only ID
          try:
              out.append((int(p), "FS", 0))
          except Exception:
              continue
  return out

def _topo_order(task_ids: list, preds_map: dict) -> list:
  """Kahn's algorithm for topological sort on predecessor map.
  preds_map: {id: [pred_ids...]}
  Returns a list of task ids. If cycle, returns a best-effort order (sources first) and leaves others at the end.
  """
  from collections import defaultdict, deque
  # Build indegree graph
  indeg = {t: 0 for t in task_ids}
  succ = defaultdict(list)
  for t in task_ids:
      for (p, _typ, _lag) in preds_map.get(t, []):
          if p in indeg:
              indeg[t] += 1
              succ[p].append(t)
  q = deque([t for t, d in indeg.items() if d == 0])
  order = []
  while q:
      n = q.popleft()
      order.append(n)
      for m in succ.get(n, []):
          indeg[m] -= 1
          if indeg[m] == 0:
              q.append(m)
  # If not all tasks were ordered, append the remaining to avoid crash
  if len(order) < len(task_ids):
      leftovers = [t for t in task_ids if t not in order]
      order.extend(leftovers)
  return order

# ------------------------------------------------------------------
# Schedule computation
# ------------------------------------------------------------------
def compute_schedule(df: pd.DataFrame, j0_datetime: datetime, reverse_to_j0: bool = True) -> pd.DataFrame:
    # Advanced dependencies + PERT calculation is the only logic retained.
    gantt_df = df.copy(deep=True)

    if {"Task_ID", "Predecessors"}.issubset(gantt_df.columns):
        gantt_df["Task_ID"] = pd.to_numeric(gantt_df["Task_ID"], errors="coerce")
        dur_map = {}
        for tid, sub in gantt_df.groupby("Task_ID"):
            if pd.isna(tid):
                continue
            d = _duration_days(sub.iloc[0])
            dur_map[int(tid)] = max(0, int(d))

        preds = {}
        for tid, sub in gantt_df.groupby("Task_ID"):
            if pd.isna(tid):
                continue
            p = sub.iloc[0].get("Predecessors", "")
            preds[int(tid)] = _parse_predecessors(p)

        task_ids = [int(t) for t in sorted(dur_map.keys())]
        order = _topo_order(task_ids, preds)

        ES = {t: None for t in task_ids}
        EF = {t: None for t in task_ids}
        project_start = j0_datetime - timedelta(days=365)

        for t in order:
            dur = timedelta(days=dur_map.get(t, 0))
            es_candidate = project_start
            constraints = []
            for (p, typ, lag_days) in preds.get(t, []):
                if p not in EF or ES.get(p) is None:
                    continue
                lag = timedelta(days=int(lag_days))
                if typ == "FS":
                    constraints.append(EF[p] + lag)
                elif typ == "SS":
                    constraints.append(ES[p] + lag)
                elif typ == "FF":
                    constraints.append(EF[p] + lag - dur)
                elif typ == "SF":
                    constraints.append(ES[p] + lag - dur)
                else:
                    constraints.append(EF[p] + lag)
            if constraints:
                es_candidate = max([c for c in constraints if c is not None] + [project_start])
            ES[t] = es_candidate
            EF[t] = es_candidate + dur

        gantt_df["Start"] = pd.NaT
        gantt_df["Finish"] = pd.NaT
        for t in task_ids:
            mask = gantt_df["Task_ID"].astype("Int64") == t
            if ES.get(t) is not None and EF.get(t) is not None:
                gantt_df.loc[mask, "Start"] = ES[t]
                gantt_df.loc[mask, "Finish"] = EF[t]

        if reverse_to_j0 and pd.notna(gantt_df["Finish"]).any():
            max_finish = pd.to_datetime(gantt_df["Finish"]).max()
            if pd.notna(max_finish):
                delta = j0_datetime - max_finish
                gantt_df["Start"] = pd.to_datetime(gantt_df["Start"]) + delta
                gantt_df["Finish"] = pd.to_datetime(gantt_df["Finish"]) + delta

        return gantt_df

    # No legacy fallback: return as-is if not schedulable
    gantt_df["Start"] = pd.NaT
    gantt_df["Finish"] = pd.NaT
    return gantt_df

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

        # Task dropdown (single select), filtered by selected roles if any
        st.markdown("**Task filter**")
        if sel_roles and "Role" in gantt_df.columns:
            task_base = gantt_df[gantt_df["Role"].astype(str).isin(sel_roles)]
        else:
            task_base = gantt_df

        if "Document" in task_base.columns:
            task_values = (
                sorted(task_base["Document"].dropna().astype(str).unique().tolist())
                if not task_base["Document"].dropna().empty else []
            )
        else:
            task_values = []

        task_options = ["All tasks"] + task_values if task_values else ["All tasks"]
        selected_task = st.selectbox("Task to display", options=task_options, index=0)

    # Apply Gantt filters
    filt = gantt_df.copy()

    # Role filter (if Role column exists and user selected roles)
    if "Role" in filt.columns and sel_roles:
        filt = filt[filt["Role"].astype(str).isin([str(r) for r in sel_roles])]

    # Task filter (single selection)
    if selected_task != "All tasks":
        filt = filt[filt["Document"].astype(str) == str(selected_task)]
        

    # ----- Gantt: one bar per task + chronological order -----
    if filt.empty or filt["Start"].isna().all() or filt["Finish"].isna().all():
        st.warning("⚠️ No schedulable data after filters/dates.")
        return

    # Deduplicate by task so duration is counted once
    dedup_keys = [c for c in ["Step_No", "Document"] if c in filt.columns]
    gantt_plot = (
        filt.sort_values(dedup_keys)
            .drop_duplicates(subset=dedup_keys, keep="first")
            .copy()
        if dedup_keys else filt.copy()
    )

    # Ensure datetimes and sort chronologically
    gantt_plot["Start"] = pd.to_datetime(gantt_plot["Start"], errors="coerce")
    gantt_plot["Finish"] = pd.to_datetime(gantt_plot["Finish"], errors="coerce")
    sort_cols = [c for c in ["Start", "Finish", "Concurrency_Group", "Step_No", "Document"] if c in gantt_plot.columns]
    gantt_plot = gantt_plot.sort_values(by=sort_cols, ascending=[True]*len(sort_cols), na_position="last").reset_index(drop=True)

    # Color by concurrency group (keep original look of thick lines)
    color_map = {
        "1": "#2ecc40", "2": "#0074d9", "3": "#ff851b", "4": "#b10dc9", "5": "#ff4136",
        "6": "#7fdbff", "7": "#3d9970", "8": "#f012be", "9": "#85144b", "10": "#aaaaaa",
    }
        # Couleurs fixes pour les équipes de Regulatory Affairs
    RA_COLORS = {
        "Regulatory Strategist (Global/EU)": "#1f77b4",  # bleu
        "Regulatory CMC": "#ff7f0e",                    # orange
        "Labeling": "#2ca02c",                          # vert
        "Regulatory Operations": "#d62728",             # rouge
        "Local Affiliates (Japan / China / Canada / EU )": "#9467bd",  # violet
        "Global Regulatory Lead": "#8c564b",            # marron
    }
    

    fig = Figure()

    # Build a fixed color per Role (all roles are RA). Use RA_COLORS when available, else fall back to a palette.
    palette = getattr(px.colors.qualitative, "Plotly", [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    unique_roles = [r for r in gantt_plot["Role"].dropna().astype(str).unique()]
    role_colors = {}
    for idx, r in enumerate(unique_roles):
        base_col = RA_COLORS.get(r) if 'RA_COLORS' in globals() else None
        role_colors[r] = base_col if base_col else palette[idx % len(palette)]

    # Track which roles already appear in the legend to avoid duplicates
    seen_roles = set()

    for i, (_, row) in enumerate(gantt_plot.iterrows()):
        dept = str(row.get("Department", "")).strip()
        role = str(row.get("Role", "")).strip()
        group_key = str(row.get("Concurrency_Group", ""))

        # One fixed color per Role (no department filtering)
        chosen_color = role_colors.get(role, "#888888")

        # Legend: one entry per Role (first occurrence only)
        legend_name = role if role else row["Document"]
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
                f"<b>{row['Document']}</b><br>"
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

    fig.update_yaxes(
        tickvals=list(range(len(gantt_plot))),
        ticktext=gantt_plot["Document"],
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
    min_start = pd.to_datetime(filt["Start"]).min()
    max_finish = pd.to_datetime(filt["Finish"]).max()
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
            r"$\mathrm{FTE}_{task} = \dfrac{\mathrm{Effort\\_Days}}{\mathrm{Working\\_days\\_per\\_FTE}}$"
        )

        calc_method = st.selectbox(
            "Calculation method",
            options=["Annualized FTE (Effort_Days / Working days per FTE)"],
            index=0,
            help="Annualizes each task effort to an FTE fraction over a full-time year."
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
            task_view_cols = [c for c in ["Role", "Department", "Document", "Effort_Days", "Start", "Finish"] if c in filt.columns]
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
                st.info("`data/HR_Capacity.csv` not found. Capacity comparison will be skipped.")

            # Aggregate per role
            agg_specs = {
                "Document": "nunique",
                "Effort_Days": "sum",
                "FTE_Task": "sum",
            }
            if "Cost" in task_view.columns:
                agg_specs["Cost"] = "sum"

            role_agg = (
                task_view.groupby("Role", as_index=False)
                .agg(**{
                    "Tasks": ("Document", "nunique"),
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
            show_task_cols = [c for c in ["Role", "Department", "Document", "Effort_Days", "FTE_Task", "Start", "Finish"] if c in task_view.columns]
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

            st.download_button(
                "📥 Download FTE per Task (CSV)",
                data=task_view[show_task_cols].to_csv(index=False),
                file_name=f"FTE_per_Task_{safe_process}_{date_str}.csv",
                mime="text/csv",
                key="dl_fte_task"
            )

            st.download_button(
                "📥 Download FTE by Role (CSV)",
                data=role_agg[display_cols].to_csv(index=False),
                file_name=f"FTE_by_Role_{safe_process}_{date_str}.csv",
                mime="text/csv",
                key="dl_fte_role"
            )

    # ----- Critical Path (task-level, deduplicated, chronological) -----
    with st.expander("🔥 Show Critical Path Tasks", expanded=False):
        st.markdown("Les tâches critiques sont celles qui n'ont **aucune marge de retard (Slack = 0)**.")

        # Build task-level view to avoid duplicates
        grp_keys = ["Concurrency_Group"]
        if "Step_No" in filt.columns: grp_keys.append("Step_No")
        if "Document" in filt.columns: grp_keys.append("Document")

        if not {"Concurrency_Group", "Document"}.issubset(filt.columns):
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

            critical_tasks = task_level[task_level["Critical"]].copy()

            if critical_tasks.empty:
                st.info("✅ Aucune tâche critique détectée (toutes ont une marge suffisante).")
            else:
                out = critical_tasks.rename(columns={
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

                sort_cols = [c for c in ["Start", "Finish", "Group", "Step", "Document"] if c in out.columns]
                out = out.sort_values(by=sort_cols, ascending=[True]*len(sort_cols), na_position="last")

                # Pretty date display
                if "Start" in out.columns:
                    out["Start"] = out["Start"].dt.strftime("%Y-%m-%d")
                if "Finish" in out.columns:
                    out["Finish"] = out["Finish"].dt.strftime("%Y-%m-%d")

                display_cols = [c for c in ["Step","Document","Duration (Days)","Effort (FTE Days)","Start","Finish","Group"] if c in out.columns]
                st.dataframe(out[display_cols].reset_index(drop=True), use_container_width=True)

                # CSV export
                date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_process = re.sub(r"[^A-Za-z0-9]+", "_", process).strip("_")
                csv_critical = out[display_cols].to_csv(index=False)
                st.download_button(
                    label="📥 Download Critical Path (CSV)",
                    data=csv_critical,
                    file_name=f"Critical_Path_{safe_process}_{date_str}.csv",
                    mime="text/csv",
                    key="download_critical_tasks"
                )

if __name__ == "__main__":
    main()
    
    
    
    
