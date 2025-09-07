import streamlit as st
import pandas as pd
import io
import contextlib
import traceback
from io import StringIO

# ---------------------------
# Helper: process_csv_data
# ---------------------------
def process_csv_data(df_raw=None):
    try:
        if df_raw is not None:
            if isinstance(df_raw, (str, bytes)):
                df_raw = pd.read_csv(df_raw, delimiter=';', header=None, dtype=str, skiprows=1)

        raid_rows_raw = df_raw[df_raw.iloc[:, 0].astype(str).str.strip().str.startswith("Raid ")]
        print(f"Total 'Raid ' rows in raw file: {len(raid_rows_raw)}")

        header_row_idx = df_raw[df_raw.iloc[:, 0].astype(str).str.strip() == "Name"].index
        if header_row_idx.empty:
            print("❌ Could not find a row strictly equal to 'Name'.")
            return None, "Could not find a row strictly equal to 'Name'."
        header_row_idx = header_row_idx[0]

        df = df_raw.copy()
        df.columns = df.iloc[header_row_idx].astype(str).str.strip()
        df = df.iloc[header_row_idx + 1:].reset_index(drop=True)
        df = df[df.iloc[:, 0].astype(str).str.strip().str.startswith("Raid ")].reset_index(drop=True)

        if df.empty:
            print("❌ No rows found strictly starting with 'Raid '.")
            return None, "No rows found strictly starting with 'Raid '."

        new_col = [
            'Name','Time','Start','Stop','Team','Player','Raid 1','Raid 2','Raid 3',
            'D1','D2','D3','D4','D5','D6','D7','Successful','Empty','Unsuccessful',
            'Bonus','No Bonus','Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9',
            'RT0','RT1','RT2','RT3','RT4','RT5','RT6','RT7','RT8','RT9',
            'DT0','DT1','DT2','DT3','DT4','Hand touch','Running hand touch','Toe touch','Running Kick',
            'Reverse Kick','Side Kick','Defender self out (lobby, shirt pull)','Body hold',
            'Ankle hold','Thigh hold','Push','Dive','DS0','DS1','DS2','DS3','In Turn',
            'Out Turn','Create Gap','Jump','Dubki','Struggle','Release','Block','Chain_def','Follow',
            'Technical Point','All Out','RL1','RL2','RL3','RL4','RL5','RL6','RL7','RL8','RL9','RL10',
            'RL11','RL12','RL13','RL14','RL15','RL16','RL17','RL18','RL19','RL20','RL21','RL22','RL23',
            'RL24','RL25','RL26','RL27','RL28','RL29','RL30',
            'Raider self out (lobby, time out, empty raid 3)'
        ]

        if len(df.columns) == len(new_col):
            df.columns = new_col
        else:
            print(f"❌ Column mismatch: got {len(df.columns)}, expected {len(new_col)}")
            return None, f"Column mismatch: got {len(df.columns)}, expected {len(new_col)}"

        return df.copy(), "Processing complete. Cleaned DataFrame returned."

    except Exception as e:
        msg = f"An unexpected error occurred: {e}"
        print(msg)
        traceback.print_exc()
        return None, msg

# ---------------------------
# Helper: process_and_qc
# ---------------------------
def process_and_qc(df_in):
    buf = StringIO()
    qc_messages = []

    try:
        with contextlib.redirect_stdout(buf):
            df = df_in.copy()

            # ---------------- Initial Setup ----------------
            tour_id = "T001"
            seas_id = "PKL-12"
            match_no = "02"
            match_id = 6465

            # ---------------- Drop unused columns ----------------
            df.drop(['Time', 'Team'], axis=1, inplace=True, errors='ignore')

            # ---------------- Raid_Number ----------------
            if 'Raid 1' in df.columns:
                df['Raid 1'] = df['Raid 1'].fillna(0)
            if 'Raid 2' in df.columns:
                df['Raid 2'] = df['Raid 2'].replace(1, 2)
                df['Raid 2'] = df['Raid 2'].fillna(0)
            else:
                df['Raid 2'] = 0
            if 'Raid 3' in df.columns:
                df['Raid 3'] = df['Raid 3'].replace(1, 3)
                df['Raid 3'] = df['Raid 3'].fillna(0)
            else:
                df['Raid 3'] = 0

            df['Raid_1'] = (
                df['Raid 1'].fillna(0).astype(int) +
                df['Raid 2'].fillna(0).astype(int) +
                df['Raid 3'].fillna(0).astype(int)
            )
            df = df.drop(['Raid 2', 'Raid 3'], axis=1).rename(columns={
                'Raid_1': 'Raid_Number',
                'Name': 'Event_Number',
                'Technical Point': 'Technical_Point',
                'All Out': 'All_Out'
            })

            # ---------------- Number_of_Defenders ----------------
            cols = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
            for idx, col in enumerate(cols, 1):
                if col in df.columns:
                    df[col] = df[col].replace(1, idx)
            df['Number_of_Defenders'] = df[[c for c in cols if c in df.columns]].fillna(0).sum(axis=1).astype(int)
            df.drop(columns=[c for c in cols if c in df.columns], inplace=True, errors='ignore')

            # ---------------- Outcome ----------------
            if 'Successful' in df.columns:
                df['Successful'] = df['Successful'].replace({1: 'Successful', 0: ''})
            if 'Empty' in df.columns:
                df['Empty'] = df['Empty'].replace({1: 'Empty', 0: ''})
            if 'Unsuccessful' in df.columns:
                df['Unsuccessful'] = df['Unsuccessful'].replace({1: 'Unsuccessful', 0: ''})
            exist_outcome_cols = [c for c in ['Successful', 'Empty', 'Unsuccessful'] if c in df.columns]
            if exist_outcome_cols:
                df['Outcome'] = df[exist_outcome_cols].fillna('').sum(axis=1)
                df.drop([c for c in exist_outcome_cols], axis=1, inplace=True, errors='ignore')

            # ---------------- Bonus ----------------
            if 'Bonus' in df.columns:
                df['Bonus'] = df['Bonus'].replace({1: "Yes", 0: ''})
            if 'No Bonus' in df.columns:
                df['No Bonus'] = df['No Bonus'].replace({1: "No", 0: ''})
                df['Bonus'] = (df.get('Bonus', '').astype(str) + ' ' + df['No Bonus'].astype(str)).str.strip()
                df.drop(['No Bonus'], axis=1, inplace=True, errors='ignore')

            # ---------------- Zone_of_Action ----------------
            zone_cols = ['Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9']
            for col in zone_cols:
                if col in df.columns:
                    df[col] = df[col].replace({1: col, 0: ""})
            df['Zone_of_Action'] = df[[c for c in zone_cols if c in df.columns]].sum(axis=1) if any(c in df.columns for c in zone_cols) else ""
            df.drop(columns=[c for c in zone_cols if c in df.columns], inplace=True, errors='ignore')

            # ---------------- Raiding_Team_Points ----------------
            cols_rt = [f'RT{i}' for i in range(10)]
            for col in cols_rt:
                if col in df.columns:
                    num = int(col.replace("RT", ""))
                    df[col] = df[col].replace(1, num)
            if any(c in df.columns for c in cols_rt):
                df['Raiding_Team_Points'] = df[[c for c in cols_rt if c in df.columns]].fillna(0).sum(axis=1).astype(int)
            else:
                df['Raiding_Team_Points'] = 0
            df.drop(columns=[c for c in cols_rt if c in df.columns], inplace=True, errors='ignore')

            # ---------------- Defending_Team_Points ----------------
            cols_dt = [f'DT{i}' for i in range(5)]
            for col in cols_dt:
                if col in df.columns:
                    num = int(col.replace('DT', ''))
                    df[col] = df[col].replace(1, num)
            if any(c in df.columns for c in cols_dt):
                df['Defending_Team_Points'] = df[[c for c in cols_dt if c in df.columns]].fillna(0).astype(int).sum(axis=1)
            else:
                df['Defending_Team_Points'] = 0
            df.drop(columns=[c for c in cols_dt if c in df.columns], inplace=True, errors='ignore')

            # ---------------- Attacking_Skill ----------------
            atk_cols = [
                'Hand touch', 'Running hand touch', 'Toe touch', 'Running Kick',
                'Reverse Kick', 'Side Kick', 'Defender self out (lobby, shirt pull)'
            ]
            for col in atk_cols:
                if col in df.columns:
                    df[col] = df[col].replace({1: col, 0: ''})
            if any(c in df.columns for c in atk_cols):
                df[atk_cols] = df[[c for c in atk_cols if c in df.columns]].fillna('')
                df['Attacking_Skill'] = df[[c for c in atk_cols if c in df.columns]].apply(lambda x: ', '.join(filter(None, x)), axis=1)
                df.drop(columns=[c for c in atk_cols if c in df.columns], inplace=True, errors='ignore')
            else:
                df['Attacking_Skill'] = ''

            # ---------------- Defensive_Skill ----------------
            def_cols = [
                'Body hold', 'Ankle hold', 'Thigh hold', 'Push', 'Dive', 'Block',
                'Chain_def', 'Follow', 'Raider self out (lobby, time out, empty raid 3)'
            ]
            for col in def_cols:
                if col in df.columns:
                    df[col] = df[col].replace({1: col, 0: ''})
            if any(c in df.columns for c in def_cols):
                df[def_cols] = df[[c for c in def_cols if c in df.columns]].fillna('')
                df['Defensive_Skill'] = df[[c for c in def_cols if c in df.columns]].apply(lambda x: ', '.join(filter(None, x)), axis=1)
                df.drop(columns=[c for c in def_cols if c in df.columns], inplace=True, errors='ignore')
            else:
                df['Defensive_Skill'] = ''

            # ---------------- Defenders_Self_Out ----------------
            ds_cols = ['DS0','DS1','DS2','DS3']
            for col in ds_cols:
                if col in df.columns:
                    num = int(col.replace('DS', ''))
                    df[col] = df[col].replace(1, num)
            if any(c in df.columns for c in ds_cols):
                df['No_of_Defenders_Self_Out'] = df[[c for c in ds_cols if c in df.columns]].fillna(0).astype(int).sum(axis=1)
                df.drop(columns=[c for c in ds_cols if c in df.columns], inplace=True, errors='ignore')
            else:
                df['No_of_Defenders_Self_Out'] = 0

            # ---------------- Counter_Action_Skill ----------------
            cas_cols = ['In Turn', 'Out Turn', 'Create Gap', 'Jump', 'Dubki', 'Struggle', 'Release']
            for col in cas_cols:
                if col in df.columns:
                    df[col] = df[col].replace({1: col, 0: ''})
            if any(c in df.columns for c in cas_cols):
                df[cas_cols] = df[[c for c in cas_cols if c in df.columns]].fillna('')
                df['Counter_Action_Skill'] = df[[c for c in cas_cols if c in df.columns]].apply(lambda x: ', '.join(filter(None, x)), axis=1)
                df.drop(columns=[c for c in cas_cols if c in df.columns], inplace=True, errors='ignore')
            else:
                df['Counter_Action_Skill'] = ''

            # ---------------- Raid_Length ----------------
            rl_cols = [f'RL{i}' for i in range(1, 31)]
            for col in rl_cols:
                if col in df.columns:
                    num = int(col.replace('RL', ''))
                    df[col] = df[col].replace(1, num)
            if any(c in df.columns for c in rl_cols):
                df['Raid_Length'] = 30 - df[[c for c in rl_cols if c in df.columns]].fillna(0).astype(int).sum(axis=1)
                df.drop(columns=[c for c in rl_cols if c in df.columns], inplace=True, errors='ignore')
            else:
                df['Raid_Length'] = 0

            # ---------------- Add Identifiers ----------------
            n = len(df)
            df['Tournament_ID'] = tour_id
            df['Season_ID'] = seas_id
            df['Match_No'] = match_no
            df['Match_ID'] = match_id
            df['Match_Raid_No'] = range(1, n + 1)

            # ---------------- Raider & Defenders Names ----------------
            if 'Player' in df.columns:
                parts = df['Player'].str.split(r'\s*\|\s*', expand=True)
                names = parts.apply(lambda s: s.str.split('-', n=1).str[1].str.strip().str.title())
                while names.shape[1] < 8:
                    names[names.shape[1]] = None
                names = names.iloc[:, :8].rename(columns={
                    0: 'Raider_Name', 1: 'Defender_1', 2: 'Defender_2',
                    3: 'Defender_3', 4: 'Defender_4', 5: 'Defender_5',
                    6: 'Defender_6', 7: 'Defender_7'
                })
                df = df.drop(columns='Player').join(names)
            else:
                # If no Player column, add empty defender columns
                for i in range(1, 8):
                    df[f'Defender_{i}'] = None
                df['Raider_Name'] = None

            # ---------------- Start & Stop Time ----------------
            if 'Start' in df.columns:
                df['Start'] = df['Start'].str.split(',').str[0]
            if 'Stop' in df.columns:
                df['Stop'] = df['Stop'].str.split(',').str[0]

            def parse_time(t):
                if pd.isna(t):
                    return pd.Timedelta(seconds=0)
                parts = list(map(int, str(t).split(":")))
                return pd.Timedelta(minutes=parts[0], seconds=parts[1]) if len(parts) == 2 else pd.Timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])

            df['start_td'] = df['Start'].apply(parse_time) if 'Start' in df.columns else pd.Timedelta(seconds=0)
            df['stop_td'] = df['Stop'].apply(parse_time) if 'Stop' in df.columns else pd.Timedelta(seconds=0)
            if 'start_td' in df.columns and 'stop_td' in df.columns:
                df['Time'] = (df['stop_td'] - df['start_td']).dt.total_seconds().apply(lambda x: f"{int(x//60):02}:{int(x%60):02}")
                df.drop(columns=['start_td', 'stop_td', 'Stop', 'Start'], inplace=True, errors='ignore')
            else:
                df['Time'] = None

            # ---------------- Add Default Columns ----------------
            new_columns = [
                'Video_Link', 'Video', 'Event', 'YC_Extra', 'Team_Raid_Number',
                'Raiding_Team_ID', 'Raiding_Team_Name', 'Defending_Team_ID', 'Defending_Team_Name',
                'Player_ID', 'Raider_ID', 'Raiding_Team_Points_Pre', 'Defending_Team_Points_Pre',
                'Raiding_Touch_Points', 'Raiding_Bonus_Points', 'Raiding_Self_Out_Points',
                'Raiding_All_Out_Points', 'Defending_Capture_Points', 'Defending_Bonus_Points',
                'Defending_Self_Out_Points', 'Defending_All_Out_Points', 'Number_of_Raiders',
                'Raider_Self_Out', 'Defenders_Touched_or_Caught', 'Half'
            ]
            for col in new_columns:
                if col not in df.columns:
                    df[col] = None

            # ---------------- Final Column Order ----------------
            final_cols = [
                "Season_ID", "Tournament_ID", "Match_No", "Match_ID", "Event_Number", "Match_Raid_No",
                "Team_Raid_Number", "Raid_Number", "Half", "Time", "Raid_Length", "Outcome", "All_Out",
                "Bonus", "Technical_Point", 'Raider_Self_Out', "Raiding_Touch_Points", "Raiding_Bonus_Points",
                "Raiding_Self_Out_Points", "Raiding_All_Out_Points", "Raiding_Team_Points",
                "Defending_Capture_Points", "Defending_Bonus_Points", "Defending_Self_Out_Points",
                "Defending_All_Out_Points", "Defending_Team_Points", "Number_of_Raiders",
                "Defenders_Touched_or_Caught", "Raiding_Team_Points_Pre", "Defending_Team_Points_Pre",
                "Zone_of_Action", "Raider_Name", "Player_ID", "Raider_ID", "Raiding_Team_ID",
                "Raiding_Team_Name", "Defending_Team_ID", "Defending_Team_Name", "Number_of_Defenders",
                "Defender_1", "Defender_2", "Defender_3", "Defender_4", "Defender_5", "Defender_6",
                "Defender_7", "No_of_Defenders_Self_Out", "Attacking_Skill", "Defensive_Skill",
                "Counter_Action_Skill", "Video_Link", "Video", "Event", "YC_Extra"
            ]
            # add missing final columns if not in df, but keep order
            for c in final_cols:
                if c not in df.columns:
                    df[c] = None
            df = df[final_cols]

            # ---------------- Points Calculation ----------------
            df["Raiding_Bonus_Points"] = (df["Bonus"] == "Yes").astype(int)
            defender_cols = ['Defender_1', 'Defender_2', 'Defender_3', 'Defender_4', 'Defender_5', 'Defender_6', 'Defender_7']
            df['Raiding_Touch_Points'] = 0
            mask = df['Outcome'] == 'Successful'
            if mask.any():
                df.loc[mask, 'Raiding_Touch_Points'] = df.loc[mask, defender_cols].notna().sum(axis=1) - df.loc[mask, 'No_of_Defenders_Self_Out'].fillna(0)
            df["Raiding_All_Out_Points"] = (((df['Outcome'] == 'Successful') & (df["All_Out"] == 1)).astype(int) * 2)
            df['Raiding_Self_Out_Points'] = df['No_of_Defenders_Self_Out']
            df['Defending_Bonus_Points'] = (((df['Number_of_Defenders'] <= 3) & (df['Outcome'] == 'Unsuccessful')).astype(int))
            df["Raider_Self_Out"] = (df["Defensive_Skill"] == "Raider self out (lobby, time out, empty raid 3)").astype(int)
            df['Defending_Capture_Points'] = (((df['Outcome'] == 'Unsuccessful') & (df['Raider_Self_Out'] == 0)).astype(int))
            df["Defending_All_Out_Points"] = (((df['Outcome'] == 'Unsuccessful') & (df["All_Out"] == 1)).astype(int) * 2)
            df['Defending_Self_Out_Points'] = df["Raider_Self_Out"]

            # ---------------- Quality Checks ----------------
            # The following checks use print() heavily — all prints will be captured.
            # QC 1: Empty Columns
            cols_qc = ['Raid_Length', 'Outcome', 'Bonus', 'All_Out', 'Raid_Number', 'Raider_Name', 'Number_of_Defenders']
            mask = df[cols_qc].isna() | df[cols_qc].eq('')
            invalid_rows = df[mask.any(axis=1)]
            if not invalid_rows.empty:
                for idx, row in invalid_rows.iterrows():
                    empty_cols = mask.loc[idx][mask.loc[idx]].index.tolist()
                    print(f"❌ Event {row['Event_Number']}: Empty in columns → {', '.join(empty_cols)}. Please check and update.\n")
            else:
                print("✅ QC 1: All rows are completely filled. Thank you!\n")

            # QC 2: Outcome Empty consistency
            cols_qc1 = [
                'Defender_1', 'Defender_2', 'Defender_3', 'Defender_4', 'Defender_5', 'Defender_6', 'Defender_7',
                'Attacking_Skill', 'Defensive_Skill', 'Counter_Action_Skill', 'Zone_of_Action'
            ]
            cols_present = [c for c in cols_qc1 if c in df.columns]
            cols_empty_qc1 = df[cols_present].replace('', pd.NA).isna().all(axis=1) if cols_present else pd.Series([True]*len(df), index=df.index)
            mask_qc1_invalid = (
                (df['Outcome'] == 'Empty') & ~(
                    cols_empty_qc1 & 
                    (df['All_Out'] == 0) & 
                    (df['Raiding_Team_Points'] == 0) & 
                    (df['Defending_Team_Points'] == 0) & 
                    (df['Bonus'] == 'No')
                )
            )
            if mask_qc1_invalid.any():
                for idx, row in df[mask_qc1_invalid].iterrows():
                    non_empty_cols = row[cols_present].replace('', pd.NA).dropna().index.tolist()
                    print(f"❌ {row['Event_Number']}: → When Outcome is 'Empty', these columns should be empty: {', '.join(non_empty_cols)}.\n")
            else:
                print("✅ QC 2: All rows meet QC 1 conditions for Outcome = 'Empty'.\n")

            # QC 3: Successful / Unsuccessful with Bonus = No & Raider_Self_Out = 0
            cols_qc2 = ['Defender_1', 'Number_of_Defenders', 'Zone_of_Action']
            non_empty_outcomes = (
                df['Outcome'].isin(['Successful', 'Unsuccessful'])
            ) & (df['Bonus'] == 'No') & (df['Raider_Self_Out'] == 0)
            cols_filled_qc2 = df[cols_qc2].replace('', pd.NA).notna().all(axis=1)
            mask_qc2_invalid = non_empty_outcomes & ~cols_filled_qc2
            if mask_qc2_invalid.any():
                for idx, row in df[mask_qc2_invalid].iterrows():
                    empty_cols = row[cols_qc2].replace('', pd.NA).isna()
                    missing_cols = empty_cols[empty_cols].index.tolist()
                    print(f"❌ {row['Event_Number']}: When Outcome='{row['Outcome']}', Bonus='No', Raider_Self_Out=0 → Missing: {', '.join(missing_cols)}.\n")
            else:
                print("✅ QC 3: All rows are Valid\n")

            # QC 4: Raid_Number = 3 must not have Outcome 'Empty'
            mask_invalid = (df['Raid_Number'] == 3) & (df['Outcome'] == 'Empty')
            if mask_invalid.any():
                for idx, row in df[mask_invalid].iterrows():
                    print(f"❌ {row['Event_Number']}: → Outcome is 'Empty' but Raid_No = 3. Please check and update.\n")
            else:
                print("✅ QC 4: All Raid_Number = 3 rows have valid Outcomes.\n")

            # QC 5: Attacking & Defensive Points match
            def check_points(cols, total_col, label):
                print(f"\nChecking {label} → '{total_col}'\n")
                # Some cols may not exist, treat missing as zeros
                cols_present = [c for c in cols if c in df.columns]
                mismatch = df[cols_present].sum(axis=1) != df[total_col]
                if mismatch.any():
                    for idx, row in df[mismatch].iterrows():
                        expected = df.loc[idx, cols_present].sum() if cols_present else 0
                        print(f"❌ {row['Event_Number']}: → {label} mismatch (Expected: {expected}, Found: {row[total_col]})\n")
                else:
                    print(f"✅ QC 5: All rows are correct for {label}\n")

            check_points(
                ['Raiding_Touch_Points','Raiding_Bonus_Points','Raiding_Self_Out_Points','Raiding_All_Out_Points'],
                'Raiding_Team_Points',
                label="Attacking Points"
            )
            check_points(
                ['Defending_Capture_Points','Defending_Bonus_Points','Defending_Self_Out_Points','Defending_All_Out_Points'],
                'Defending_Team_Points',
                label="Defensive Points"
            )

            # QC 6: Outcome Successful/Unsuccessful must have points
            def check_points_nonzero(df_local, outcome, cols, team_name):
                outcome_mask = df_local['Outcome'].eq(outcome)
                zero_points = df_local[cols].fillna(0).sum(axis=1).eq(0)
                problem_mask = outcome_mask & zero_points
                if problem_mask.any():
                    for raid_no in df_local.loc[problem_mask, 'Event_Number'].astype(str):
                        print(f"❌ {team_name}: Raid {raid_no} — Outcome is '{outcome}', but no points were given.\n")
                else:
                    print(f"✅ QC 6: All {team_name} ({outcome}) rows are correct.\n")

            check_points_nonzero(
                df, 'Successful',
                ['Raiding_Touch_Points', 'Raiding_Bonus_Points', 'Raiding_Self_Out_Points', 'Raiding_All_Out_Points'], 'Raiding'
            )
            check_points_nonzero(
                df, 'Unsuccessful',
                ['Defending_Capture_Points', 'Defending_Bonus_Points', 'Defending_Self_Out_Points', 'Defending_All_Out_Points'], 'Defending'
            )

            # QC 7: Defending_Self_Out_Points > 1
            mismatch_df = df[df['Defending_Self_Out_Points'] > 1]
            if not mismatch_df.empty:
                for event_num in mismatch_df['Event_Number']:
                    print(f"❌ {event_num}: 'Defending_Self_Out_Points' is greater than 1. Check 'Raider self out' column.\n")
            else:
                print('✅ QC 7: All rows have correct Defending_Self_Out_Points values.\n')

            # QC 8: Successful Outcome must reset Raid_Number
            success_rows = df.index[df['Outcome'] == 'Successful']
            mismatches = []
            for idx in success_rows:
                success_event = df.loc[idx, 'Event_Number']
                check_idx = idx + 2
                if check_idx in df.index:
                    if df.loc[check_idx, 'Raid_Number'] != 1:
                        mismatches.append((success_event, df.loc[check_idx, 'Event_Number']))
            if mismatches:
                for s, c in mismatches:
                    print(f"❌ Outcome: 'Successful' {s}, --> {c} should have Raid_Number = 1.\n")
            else:
                print("✅ QC 8: All rows are correct.\n")

            # QC 9: Empty Raid Consistency
            errors_found = False
            for idx, row in df.iterrows():
                if row['Raid_Number'] == 2 and row['Outcome'] == 'Empty':
                    if idx >= 2:
                        prev_row = df.loc[idx - 2]
                        if prev_row['Raid_Number'] == 1 and prev_row['Outcome'] != 'Empty':
                            print(f"❌ {row['Event_Number']}: Previous raid not Empty\n")
                            errors_found = True
            if not errors_found:
                print("✅ QC 9: All rows are correct.\n")

            # QC 10: Raid_Length should be > 2
            errors_found = False
            for idx, row in df.iterrows():
                try:
                    rl_val = float(row['Raid_Length'])
                except Exception:
                    rl_val = 0
                if rl_val <= 2:
                    print(f"⚠️ {row['Event_Number']}: Raid_Length is {row['Raid_Length']}\n")
                    errors_found = True
            if not errors_found:
                print("✅ QC 10: All rows have valid Raid_Length values.\n")

            # QC 11: Successful, No Bonus -> Defensive & Counter Action Skill consistency
            filtered_df = df[
                (df['Outcome'] == 'Successful') &
                (df['Bonus'] == 'No') &
                (df['Raiding_Touch_Points'].fillna(0) > 0)
            ]
            mismatched_events = filtered_df.loc[
                (filtered_df['Defensive_Skill'].replace('', pd.NA).isna()) !=
                (filtered_df['Counter_Action_Skill'].replace('', pd.NA).isna()),
                'Event_Number'
            ]
            if not mismatched_events.empty:
                for event_num in mismatched_events:
                    print(f"❌ {event_num}: -'Defensive_Skill' or 'Counter_Action_Skill' missing.\n")
            else:
                print("✅ QC 11: All rows are correct\n")

            # QC 12: Successful, No Bonus, No Defenders Self Out
            fil_df = df[
                (df['Outcome'] == 'Successful') &
                (df['Bonus'] == 'No') &
                (df['No_of_Defenders_Self_Out'] == 0)
            ].copy()
            for col in ['Attacking_Skill', 'Defensive_Skill', 'Counter_Action_Skill']:
                fil_df[col] = fil_df[col].replace('', pd.NA)
            cond1 = (fil_df['Attacking_Skill'].isna() &
                     (fil_df['Defensive_Skill'].isna() | fil_df['Counter_Action_Skill'].isna()))
            cond2 = (fil_df['Attacking_Skill'].notna() &
                     (fil_df['Defensive_Skill'].notna() | fil_df['Counter_Action_Skill'].notna()))
            qc_wrong_rows = fil_df.loc[cond1 | cond2, 'Event_Number']
            if not qc_wrong_rows.empty:
                for event in qc_wrong_rows:
                    print(f"⚠️ {event}: 'Attacking_Skill' & 'Defensive & Counter_Action_Skill' - all 3 Present Check once.\n")
            else:
                print("✅ QC 12: All rows are correct\n")

            # QC 13: Outcome = Unsuccessful -> Defensive_Skill must NOT be empty
            qc_violations = df[
                (df['Outcome'] == 'Unsuccessful') &
                (df['Defensive_Skill'].isna() | (df['Defensive_Skill'].str.strip() == ''))
            ]
            if not qc_violations.empty:
                for idx, row in qc_violations.iterrows():
                    print(f"❌ {row['Event_Number']}: Outcome is 'Unsuccessful' and 'Defensive_Skill' is empty.\n")
            else:
                print("✅ Qc 13: All rows are correct\n")

            # QC 14: Checking Raid_numbers
            def kabaddi_raid_number_qc_grouped(df):
                """
                QC to validate Raid_Number using Match_Raid_Number for sorting.
                Prints custom messages for mismatches or a success message if no errors.
            
                Required columns:
                ['Match_Raid_No', 'Event_Number', 'Outcome', 'Raiding_Team_Name', 'Raid_Number']
                """
                # Sort by Match_Raid_No to ensure chronological order
                df = df.sort_values(by='Match_Raid_No').reset_index(drop=True)
            
                grouped = df.groupby('Raiding_Team_Name')
                error_found = False  # Flag to check if any errors occur
            
                for team, team_df in grouped:
                    empty_count = 0  # Track consecutive empty raids for this team
            
                    for _, row in team_df.iterrows():
                        outcome = row['Outcome']
                        raid_num = row['Raid_Number']
                        event_number = row['Event_Number']
            
                        # ---- Determine expected Raid_Number ----
                        if empty_count == 0:
                            expected = 1
                        elif empty_count == 1:
                            expected = 2
                        else:
                            expected = 3  # Do-or-Die
            
                        # ---- Check for errors ----
                        if raid_num != expected:
                            print(
                                f"❌ {event_number}: Outcome is '{outcome}' and 'Raid_Number' is {raid_num}. "
                                f"Expected 'Raid_Number': {expected}. Please check and update.\n"
                            )
                            error_found = True
            
                        # ---- Update empty_count ----
                        if outcome == "Empty":
                            empty_count += 1
                        else:
                            empty_count = 0  # Reset on Successful or Unsuccessful
            
                # Final message if no errors found
                if not error_found:
                    print("✅ QC 14: All rows are correct\n")

            kabaddi_raid_number_qc_grouped(df)


        content = buf.getvalue()
        qc_messages = content.splitlines()
        print(content)
        return df, qc_messages

    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error during processing: {e}")
        print(tb)
        return None, [f"Error during processing: {e}", tb]


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide", page_title="Kabaddi Data Tool")

# ---------------------------
# Session State Reset
# ---------------------------
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'qc_results' not in st.session_state:
    st.session_state.qc_results = None

def reset_state():
    st.session_state.processed_df = None
    st.session_state.qc_results = None

def reset_on_mode_change():
    reset_state()

# ---------------------------
# Mode Selection
# ---------------------------
mode = st.radio(
    "Choose mode",
    ["Cleaning Tool", "Processing & QC Tool"],
    index=0,
    horizontal=True,
    key="mode_switch",
    on_change=reset_on_mode_change  # <-- Reset everything on mode change
)

# ---------------------------
# Cleaning Tool Mode
# ---------------------------
if mode == "Cleaning Tool":
    st.title("Kabaddi Data Cleaning Tool - Old Dashboard")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        if st.button("Process CSV", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    df_raw = pd.read_csv(uploaded_file, delimiter=';', header=None, dtype=str, skiprows=1)
                    cleaned_df, message = process_csv_data(df_raw)
                    if cleaned_df is not None:
                        st.success(message)
                        st.session_state.processed_df = cleaned_df
                    else:
                        st.error(message)
                        st.session_state.processed_df = None
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    st.session_state.processed_df = None

    if st.session_state.processed_df is not None:
        st.header("Processed Data Preview")

        # Safely get rows and columns
        rows, cols = st.session_state.processed_df.shape if st.session_state.processed_df is not None else (0, 0)
        st.write(f"**Total rows:** `{rows}` | **Total columns:** `{cols}`")
    
        st.dataframe(st.session_state.processed_df.head())

        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(st.session_state.processed_df)

        if st.download_button(
            "Download Cleaned CSV",
            use_container_width=True,
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv"
        ):
            reset_state()


# ---------------------------
# Processing & QC Tool Mode
# ---------------------------
else:
    st.title("Kabaddi Data Processing & QC Tool - Old Dashboard")
    uploaded_file = st.file_uploader("Upload CSV", type="csv", on_change=reset_state)

    if uploaded_file:
        try:
            df_original = pd.read_csv(uploaded_file)
        except Exception:
            df_original = pd.read_csv(uploaded_file, delimiter=';', header=None, dtype=str)

        st.subheader("Preview Uploaded Data")

        # Safely get rows and columns
        rows, cols = df_original.shape if df_original is not None else (0, 0)
        st.write(f"**Total rows:** `{rows}` | **Total columns:** `{cols}`")
        
        st.dataframe(df_original.head())

        if st.button("Process & Run QC", use_container_width=True):
            with st.spinner("Processing data..."):
                processed_df, qc_messages = process_and_qc(df_original)
                st.session_state.processed_df = processed_df
                st.session_state.qc_results = qc_messages
                st.success("Processing and QC complete!")

    if st.session_state.qc_results:
        st.subheader("Quality Check (QC) Results")
        #st.code("\n".join(st.session_state.qc_results), language='text')

        # Make a scrollable container with fixed height
        qc_text = "\n".join(st.session_state.qc_results)  # Join first, outside f-string

        st.markdown(
            f"""
            <div style="height:300px; overflow-y:scroll; border:1px solid grey;
                    border-radius:8px; padding:10px;">
                <pre>{qc_text}</pre>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.processed_df is not None:
        st.subheader("Preview Processed Data")

        # Safely get rows and columns
        rows, cols = st.session_state.processed_df.shape
        st.write(f"**Total rows:** `{rows}` | **Total columns:** `{cols}`")

        st.dataframe(st.session_state.processed_df.head())

        # Download button only
        csv_buffer = io.StringIO()
        st.session_state.processed_df.to_csv(csv_buffer, index=False)

        # CSS to style the download button
        st.markdown(
            """
            <style>
            div.stDownloadButton>button {
                color: yellow !important;
                font-weight: bolder !important;
                font-size: 30px !important;  /* Increase font size */
                background-color: black !important;
                border: none !important;
                padding: 10px 20px !important; /* Makes button bigger */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Styled download button with original click logic
        if st.download_button(
            label="Download Processed CSV",
            data=csv_buffer.getvalue().encode('utf-8'),
            file_name="final_processed.csv",
            mime="text/csv",
            use_container_width=True
        ):
            reset_state()







