import streamlit as st
import pandas as pd
import io

# --- Mode switch (place this right after your imports, before any UI code) ---
mode = st.radio(
    "Choose mode",
    ["Cleaning Tool", "Processing & QC Tool"],
    index=0,
    horizontal=True,
    key="mode_switch",
)

if mode == "Cleaning Tool":
    def process_csv_data(df_raw):
    
        try:
            # Step 2: Find the row where the first column starts with "Name"
            header_row_idx_list = df_raw[df_raw.iloc[:, 0].astype(str).str.strip().str.startswith("Name")].index
            if header_row_idx_list.empty:
                return None, "❌ Error: Could not find a header row starting with 'Name'."
            header_row_idx = header_row_idx_list[0]

            # Step 3: Use that row as the header and keep the rows below it
            df = df_raw.copy()
            df.columns = df.iloc[header_row_idx].astype(str).str.strip()
            df = df.iloc[header_row_idx + 1:].reset_index(drop=True)

            # Step 4: Find the first and last index of rows starting with "Raid "
            raid_rows = df[df.iloc[:, 0].astype(str).str.strip().str.startswith("Raid ")].index
            if raid_rows.empty:
                return None, "❌ Error: No rows found starting with 'Raid '."

            first_raid_idx = raid_rows.min()
            last_raid_idx = raid_rows.max()

            df_between = df.loc[first_raid_idx:last_raid_idx]

            # Check for continuity
            if not df_between.iloc[:, 0].astype(str).str.strip().str.startswith("Raid ").all():
                return None, "❌ Error: Found non-'Raid' rows between the first and last 'Raid' entries."

            df = df_between

            # Step 5: Rename Columns
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

            if len(df.columns) != len(new_col):
                return None, f"❌ Error: Column count mismatch. The script expects {len(new_col)} columns, but found {len(df.columns)}."

            df.columns = new_col
            return df, "✅ Transformation complete."

        except Exception as e:
            return None, f"An unexpected error occurred: {e}"

    # --- Streamlit App UI ---

    st.set_page_config(layout="wide")

    st.title("Kabaddi Data Cleaning Tool - Old DashBoard")
    st.write("Upload your raw CSV file to clean and format it based on the predefined logic.")

    # Initialize session state
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None

    # File Uploader
    uploaded_file = st.file_uploader("1. Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Process Button
        if st.button("2. Process CSV", use_container_width=True):
            with st.spinner('Processing...'):
                try:
                    # Read the CSV file as plain text
                    df_raw = pd.read_csv(uploaded_file, delimiter=';', header=None, dtype=str, skiprows=1)
                    
                    # Process the data
                    cleaned_df, message = process_csv_data(df_raw)

                    if cleaned_df is not None:
                        st.success(message)
                        st.session_state.processed_df = cleaned_df
                    else:
                        st.error(message)
                        st.session_state.processed_df = None

                except Exception as e:
                    st.error(f"An error occurred while reading the CSV: {e}")
                    st.session_state.processed_df = None

        # Display results and download button if processing was successful
        if st.session_state.processed_df is not None:
            st.markdown("---")
            st.header("Processed Data Preview")
            
            # Show total rows and columns
            rows, cols = st.session_state.processed_df.shape
            st.write(f"**Total rows:** `{rows}` | **Total columns:** `{cols}`")
    
            st.write("**Top 5 rows:**")
            st.dataframe(st.session_state.processed_df.head())

            # Convert DataFrame to CSV string for downloading
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(st.session_state.processed_df)
            
            # Get original filename without extension
            original_filename = uploaded_file.name.rsplit('.', 1)[0]
            
            st.download_button(
                label="3. Download Cleaned CSV",
                data=csv_data,
                file_name=f'{original_filename}_cleaned.csv',
                mime='text/csv',
                use_container_width=True
            )
else:
    # --- Page Configuration ---
    st.set_page_config(page_title="Kabaddi Data QC", layout="wide")
    st.title("Kabaddi Data Processing & QC Tool - Old DashBoard")
    st.markdown("Upload your raw CSV file, run automated processing and quality checks, then download the cleaned data.")

    # --- Session State ---
    if 'qc_results' not in st.session_state:
        st.session_state.qc_results = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None

    def reset_state():
        st.session_state.qc_results = None
        st.session_state.processed_df = None

    # --- File Upload ---
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv", on_change=reset_state)

    if uploaded_file:
        df_original = pd.read_csv(uploaded_file)

        st.subheader("Preview Uploaded Data")

        # Show total rows and columns
        rows, cols = df_original.shape
        st.write(f"**Total rows:** `{rows}` | **Total columns:** `{cols}`")
       
        st.dataframe(df_original.head(5), use_container_width=True)

        if st.button("Process & Run QC"):
            with st.spinner("Processing data..."):
                try:
                    # Make a copy to avoid modifying the original dataframe in memory
                    df = df_original.copy()
                    
                    # Placeholder for QC messages
                    qc_messages = []

                    # ------------------- IDs & Metadata -------------------
                    tour_id = "T001"
                    seas_id = "pm_24-25"
                    match_no = "01"
                    match_id = "M012"

                    # Drop unnecessary columns
                    df.drop(['Time', 'Team'], axis=1, inplace=True, errors='ignore')

                    # ------------------- Raid_Number -------------------
                    df['Raid 2'] = df['Raid 2'].replace(1, 2)
                    df['Raid 3'] = df['Raid 3'].replace(1, 3)
                    df['Raid 1'] = df['Raid 1'].astype(int) + df['Raid 2'].astype(int) + df['Raid 3'].astype(int)
                    df = df.drop(['Raid 2', 'Raid 3'], axis=1).rename(columns={
                        'Raid 1': 'Raid_Number',
                        'Name': 'Event_Number',
                        'Technical Point': 'Technical_Point',
                        'All Out': 'All_Out'
                    })

                    # ------------------- Number_of_Defenders -------------------
                    cols = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
                    for idx, col in enumerate(cols, 1):
                        df[col] = df[col].replace(1, idx)
                    df['Number_of_Defenders'] = df[cols].astype(int).sum(axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- Outcome -------------------
                    df['Successful'] = df['Successful'].replace({1: 'Successful', 0: ''})
                    df['Empty'] = df['Empty'].replace({1: 'Empty', 0: ''})
                    df['Unsuccessful'] = df['Unsuccessful'].replace({1: 'Unsuccessful', 0: ''})
                    df['Outcome'] = df['Successful'] + df['Empty'] + df['Unsuccessful']
                    df.drop(['Successful', 'Unsuccessful', 'Empty'], axis=1, inplace=True)

                    # ------------------- Bonus -------------------
                    df['Bonus'] = df['Bonus'].replace({1: "Yes", 0: ''})
                    df['No Bonus'] = df['No Bonus'].replace({1: "No", 0: ''})
                    df['Bonus'] = (df['Bonus'] + df['No Bonus']).str.strip()
                    df.drop(['No Bonus'], axis=1, inplace=True)

                    # ------------------- Zone_of_Action -------------------
                    cols = ['Z1','Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9']
                    for col in cols:
                        df[col] = df[col].replace({1: col, 0: ""})
                    df['Zone_of_Action'] = df[cols].sum(axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- Raiding_Team_Points -------------------
                    cols = ['RT0','RT1','RT2','RT3','RT4','RT5','RT6','RT7','RT8','RT9']
                    for col in cols:
                        num = int(col.replace("RT",""))
                        df[col] = df[col].replace(1, num)
                    df['Raiding_Team_Points'] = df[cols].astype(int).sum(axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- Defending_Team_Points -------------------
                    cols = ['DT0','DT1','DT2','DT3','DT4']
                    for col in cols:
                        num = int(col.replace("DT",""))
                        df[col] = df[col].replace(1, num)
                    df['Defending_Team_Points'] = df[cols].astype(int).sum(axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- Attacking_Skill -------------------
                    cols = ['Hand touch','Running hand touch','Toe touch','Running Kick','Reverse Kick',
                            'Side Kick','Defender self out (lobby, shirt pull)']
                    for col in cols:
                        df[col] = df[col].replace({1: col, 0: ''})
                    df['Attacking_Skill'] = df[cols].apply(lambda x: ', '.join(filter(None, x)), axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- Defensive_Skill -------------------
                    cols = ['Body hold','Ankle hold','Thigh hold','Push','Dive','Block',
                            'Chain_def','Follow','Raider self out (lobby, time out, empty raid 3)']
                    for col in cols:
                        df[col] = df[col].replace({1: col, 0: ''})
                    df['Defensive_Skill'] = df[cols].apply(lambda x: ', '.join(filter(None, x)), axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- No_of_Defenders_Self_Out -------------------
                    cols = ['DS0','DS1','DS2','DS3']
                    for col in cols:
                        num = int(col.replace("DS",""))
                        df[col] = df[col].replace(1,num)
                    df['No_of_Defenders_Self_Out'] = df[cols].astype(int).sum(axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- Counter_Action_Skill -------------------
                    cols = ['In Turn','Out Turn','Create Gap','Jump','Dubki','Struggle','Release']
                    for col in cols:
                        df[col] = df[col].replace({1: col, 0: ''})
                    df['Counter_Action_Skill'] = df[cols].apply(lambda x: ', '.join(filter(None,x)), axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- Raiding Length -------------------
                    cols = [f'RL{i}' for i in range(1,31)]
                    for col in cols:
                        num = int(col.replace("RL",""))
                        df[col] = df[col].replace(1,num)
                    df['Raid_Length'] = 30 - df[cols].astype(int).sum(axis=1)
                    df.drop(columns=cols, inplace=True)

                    # ------------------- Match Metadata -------------------
                    n = len(df)
                    df['Tournament_ID'] = tour_id
                    df['Season_ID'] = seas_id
                    df['Match_No'] = match_no
                    df['Match_ID'] = match_id
                    df['Match_Raid_No'] = range(1, n+1)

                    # ------------------- Raider & Defenders Names -------------------
                    parts = df['Player'].str.split(r'\s*\|\s*', expand=True)
                    names = parts.apply(lambda s: s.str.split('-', n=1).str[1].str.strip())
                    needed_cols = 8
                    if names.shape[1] < needed_cols:
                        for _ in range(needed_cols - names.shape[1]):
                            names[names.shape[1]] = None
                    names = names.iloc[:, :needed_cols]
                    names.columns = ['Raider_Name','Defender_1','Defender_2','Defender_3','Defender_4','Defender_5','Defender_6','Defender_7']
                    df.drop(columns='Player', inplace=True)
                    df = df.join(names)

                    # ------------------- Start & Stop Time -------------------
                    df['Start'] = df['Start'].str.split(',').str[0]
                    df['Stop'] = df['Stop'].str.split(',').str[0]

                    def parse_time(t):
                        parts = list(map(int, t.split(":")))
                        if len(parts)==2:
                            m,s = parts
                            return pd.Timedelta(minutes=m,seconds=s)
                        elif len(parts)==3:
                            h,m,s = parts
                            return pd.Timedelta(hours=h,minutes=m,seconds=s)
                        return pd.Timedelta(seconds=0) # Handle potential errors

                    df['start_td'] = df['Start'].apply(parse_time)
                    df['stop_td'] = df['Stop'].apply(parse_time)
                    df['duration'] = df['stop_td'] - df['start_td']
                    df['total_secs'] = df['duration'].dt.total_seconds()
                    df['Time'] = df['total_secs'].apply(lambda x: f"{int(x//60):02}:{int(x%60):02}")
                    df.drop(columns=['start_td','stop_td','duration','total_secs','Stop','Start'], inplace=True)

                    # ------------------- New Columns -------------------
                    new_cols = ['Video_Link','Video','Event','YC_Extra','Team_Raid_Number',
                                'Raiding_Team_ID','Raiding_Team_Name','Defending_Team_ID','Defending_Team_Name',
                                'Player_ID','Raider_ID','Raiding_Team_Points_Pre','Defending_Team_Points_Pre',
                                'Raiding_Touch_Points','Raiding_Bonus_Points','Raiding_Self_Out_Points','Raiding_All_Out_Points',
                                'Defending_Capture_Points','Defending_Bonus_Points','Defending_Self_Out_Points','Defending_All_Out_Points',
                                'Number_of_Raiders','Raider_Self_Out','Defenders_Touched_or_Caught','Half']
                    for col in new_cols:
                        df[col] = None

                    # ------------------- Logical Column Order -------------------
                    new_order = ["Season_ID","Tournament_ID","Match_No","Match_ID","Event_Number","Match_Raid_No",
                                 "Team_Raid_Number","Raid_Number","Half","Time","Raid_Length","Outcome","All_Out","Bonus",
                                 "Technical_Point","Raider_Self_Out","Raiding_Touch_Points","Raiding_Bonus_Points",
                                 "Raiding_Self_Out_Points","Raiding_All_Out_Points","Raiding_Team_Points","Defending_Capture_Points",
                                 "Defending_Bonus_Points","Defending_Self_Out_Points","Defending_All_Out_Points","Defending_Team_Points",
                                 "Number_of_Raiders","Defenders_Touched_or_Caught","Raiding_Team_Points_Pre","Defending_Team_Points_Pre",
                                 "Zone_of_Action","Raider_Name","Player_ID","Raider_ID","Raiding_Team_ID","Raiding_Team_Name",
                                 "Defending_Team_ID","Defending_Team_Name","Number_of_Defenders","Defender_1","Defender_2","Defender_3",
                                 "Defender_4","Defender_5","Defender_6","Defender_7","No_of_Defenders_Self_Out",
                                 "Attacking_Skill","Defensive_Skill","Counter_Action_Skill",
                                 "Video_Link","Video","Event","YC_Extra"]
                    df = df[new_order]

                    # ------------------- Update Points -------------------
                    df["Raiding_Bonus_Points"] = (df["Bonus"]=="Yes").astype(int)
                    defender_cols = ['Defender_1','Defender_2','Defender_3','Defender_4','Defender_5','Defender_6','Defender_7']
                    df['Raiding_Touch_Points'] = 0
                    mask = df['Outcome']=='Successful'
                    df.loc[mask,'Raiding_Touch_Points'] = df.loc[mask,defender_cols].notna().sum(axis=1)-df.loc[mask,'No_of_Defenders_Self_Out']
                    df["Raiding_All_Out_Points"] = (((df['Outcome']=='Successful') & (df["All_Out"]==1)).astype(int)*2)
                    df['Raiding_Self_Out_Points'] = df['No_of_Defenders_Self_Out']
                    df['Defending_Bonus_Points'] = (((df['Number_of_Defenders']<=3) & (df['Outcome']=='Unsuccessful')).astype(int))
                    df["Raider_Self_Out"] = (df["Defensive_Skill"]=="Raider self out (lobby, time out, empty raid 3)").astype(int)
                    df['Defending_Capture_Points'] = (((df['Outcome']=='Unsuccessful') & (df['Raider_Self_Out']==0)).astype(int))
                    df["Defending_All_Out_Points"] = (((df['Outcome']=='Unsuccessful') & (df["All_Out"]==1)).astype(int)*2)
                    df['Defending_Self_Out_Points'] = df["Raider_Self_Out"]

                    # ------------------- Helper QC Functions -------------------
                    def replace_empty_with_na(df, cols):
                        for col in cols:
                            df[col] = df[col].replace('', pd.NA)
                        return df

                    def check_points_single(df, cols, total_col, label):
                        expected_total = df[cols].fillna(0).sum(axis=1)
                        mismatch_mask = expected_total != df[total_col]
                        for idx, row in df.loc[mismatch_mask].iterrows():
                            qc_messages.append(f"❌ {row['Event_Number']}: → {label} mismatch (Expected: {expected_total[idx]}, Found: {row[total_col]})")
                        if mismatch_mask.sum()==0:
                            qc_messages.append(f"✅ QC 5: All rows correct for {label}")

                    def check_outcome_points(df, outcome, cols, team_name):
                        outcome_mask = df['Outcome']==outcome
                        zero_points_mask = df.loc[outcome_mask, cols].fillna(0).sum(axis=1)==0
                        for event in df.loc[outcome_mask,:].loc[zero_points_mask,'Event_Number'].unique():
                            qc_messages.append(f"❌ {team_name}: {event} — Outcome = '{outcome}', no points given. Check data.")
                        if zero_points_mask.sum()==0:
                            qc_messages.append(f"✅ QC 6: All {team_name} ({outcome}) rows correct.")

                    # ------------------- QC Checks -------------------
                    # QC 1
                    qc1_cols = ['Raid_Length','Outcome','Bonus','All_Out','Raid_Number','Raider_Name','Number_of_Defenders']
                    mask_qc1 = df[qc1_cols].isna() | df[qc1_cols].eq('')
                    invalid_qc1 = df[mask_qc1.any(axis=1)]
                    for idx,row in invalid_qc1.iterrows():
                        empty_cols = mask_qc1.loc[idx][mask_qc1.loc[idx]].index.tolist()
                        qc_messages.append(f"❌ Event {row['Event_Number']}: Empty in columns → {', '.join(empty_cols)}")
                    if invalid_qc1.empty:
                        qc_messages.append("✅ QC 1: All rows are completely filled.")

                    # QC 2
                    qc2_cols = ['Defender_1','Defender_2','Defender_3','Defender_4','Defender_5','Defender_6','Defender_7',
                                'Attacking_Skill','Defensive_Skill','Counter_Action_Skill','Zone_of_Action']
                    df = replace_empty_with_na(df, qc2_cols)
                    mask_qc2_invalid = (df['Outcome']=='Empty') & ~(df[qc2_cols].isna().all(axis=1) &
                                                                    (df['All_Out']==0) &
                                                                    (df['Raiding_Team_Points']==0) &
                                                                    (df['Defending_Team_Points']==0) &
                                                                    (df['Bonus']=='No'))
                    for event in df.loc[mask_qc2_invalid,'Event_Number'].unique():
                        qc_messages.append(f"❌ {event}: Outcome='Empty', some columns not empty or points/bonus invalid.")
                    if not mask_qc2_invalid.any():
                        qc_messages.append("✅ QC 2: All Outcome='Empty' rows are valid.")

                    # QC 3
                    qc3_cols = ['Defender_1','Number_of_Defenders','Zone_of_Action']
                    mask_qc3 = df['Outcome'].isin(['Successful','Unsuccessful']) & (df['Bonus']=='No') & (df['Raider_Self_Out']==0)
                    mask_qc3_invalid = mask_qc3 & ~df[qc3_cols].notna().all(axis=1)
                    for idx,row in df.loc[mask_qc3_invalid].iterrows():
                        missing_cols = row[qc3_cols].isna()
                        missing_cols_list = missing_cols[missing_cols].index.tolist()
                        qc_messages.append(f"❌ {row['Event_Number']}: Outcome='{row['Outcome']}', Bonus='No', Raider_Self_Out=0 → Missing: {', '.join(missing_cols_list)}")
                    if not mask_qc3_invalid.any():
                        qc_messages.append("✅ QC 3: All rows are valid.")

                    # QC 4
                    mask_qc4 = (df['Raid_Number']==3) & (df['Outcome']=='Empty')
                    for event in df.loc[mask_qc4,'Event_Number'].unique():
                        qc_messages.append(f"❌ {event}: Raid_Number=3 but Outcome='Empty'. Please check.")
                    if not mask_qc4.any():
                        qc_messages.append("✅ QC 4: All Raid_Number=3 rows valid.")

                    # QC 5
                    check_points_single(df,
                                        ['Raiding_Touch_Points','Raiding_Bonus_Points','Raiding_Self_Out_Points','Raiding_All_Out_Points'],
                                        'Raiding_Team_Points','Attacking Points')
                    check_points_single(df,
                                        ['Defending_Capture_Points','Defending_Bonus_Points','Defending_Self_Out_Points','Defending_All_Out_Points'],
                                        'Defending_Team_Points','Defensive Points')

                    # QC 6
                    check_outcome_points(df,'Successful',['Raiding_Touch_Points','Raiding_Bonus_Points','Raiding_Self_Out_Points','Raiding_All_Out_Points'],'Raiding')
                    check_outcome_points(df,'Unsuccessful',['Defending_Capture_Points','Defending_Bonus_Points','Defending_Self_Out_Points','Defending_All_Out_Points'],'Defending')

                    # QC 7
                    mask_qc7 = df['Defending_Self_Out_Points']>1
                    for event in df.loc[mask_qc7,'Event_Number'].unique():
                        qc_messages.append(f"❌ {event}: Check 'Raider self out' column and update.")
                    if mask_qc7.sum()==0:
                        qc_messages.append("✅ QC 7: All rows correct.")

                    # QC 8
                    success_rows = df.index[df['Outcome']=='Successful']
                    for idx in success_rows:
                        check_idx = idx+2
                        if check_idx in df.index and df.loc[check_idx,'Raid_Number']!=1:
                            qc_messages.append(f"❌ Outcome: 'Successful' {df.loc[idx,'Event_Number']} --> {df.loc[check_idx,'Event_Number']} should have Raid_Number=1.")
                    qc_messages.append("✅ QC 8: Raid_Number checks complete.")

                    # QC 9
                    for idx,row in df.iterrows():
                        if row['Raid_Number']==2 and row['Outcome']=='Empty' and idx>=2:
                            prev_row = df.loc[idx-2]
                            if prev_row['Raid_Number']==1 and prev_row['Outcome']!='Empty':
                                qc_messages.append(f"❌ {row['Event_Number']}: Previous raid not Empty.")
                    qc_messages.append("✅ QC 9: Raid_Number sequence valid.")

                    # QC 10
                    mask_qc10 = df['Raid_Length']<=2
                    for idx,row in df.loc[mask_qc10].iterrows():
                        qc_messages.append(f"❌ {row['Event_Number']}: Raid_Length={row['Raid_Length']}")
                    if mask_qc10.sum()==0:
                        qc_messages.append("✅ QC 10: All Raid_Length values valid.")

                    # QC 11 & 12
                    qc11_df = df[(df['Outcome']=='Successful') & (df['Bonus']=='No') & (df['Raiding_Touch_Points']>0)].copy()
                    qc11_df = replace_empty_with_na(qc11_df,['Defensive_Skill','Counter_Action_Skill'])
                    qc11_df['Def_or_Counter_Missing'] = qc11_df.apply(lambda x: pd.isna(x['Defensive_Skill']) != pd.isna(x['Counter_Action_Skill']),axis=1)

                    qc12_df = df[(df['Outcome']=='Successful') & (df['Bonus']=='No') & (df['No_of_Defenders_Self_Out']==0)].copy()
                    qc12_df = replace_empty_with_na(qc12_df,['Attacking_Skill','Defensive_Skill','Counter_Action_Skill'])
                    qc12_df['Skill_Conflict'] = qc12_df.apply(lambda x: (pd.isna(x['Attacking_Skill']) & (pd.isna(x['Defensive_Skill']) | pd.isna(x['Counter_Action_Skill']))) |
                                                                    (pd.notna(x['Attacking_Skill']) & (pd.notna(x['Defensive_Skill']) | pd.notna(x['Counter_Action_Skill']))),
                                                            axis=1)

                    combined_issues = pd.concat([qc11_df.loc[qc11_df['Def_or_Counter_Missing'],'Event_Number'],
                                                qc12_df.loc[qc12_df['Skill_Conflict'],'Event_Number']]).drop_duplicates()
                    for event in combined_issues:
                        qc_messages.append(f"❌ {event}: Issue with 'Attacking_Skill' & 'Defensive/Counter_Action_Skill'.")
                    if combined_issues.empty:
                        qc_messages.append("✅ QC 11 & QC 12: All rows correct.")

                    # QC 13 Rule: When Outcome == 'Unsuccessful', Defensive_Skill must NOT be empty
                    qc_violations = df[
                        (df['Outcome'] == 'Unsuccessful') &
                        (df['Defensive_Skill'].isna() | df['Defensive_Skill'].fillna('').str.strip().eq(''))
                    ]
                    
                    # Show the violations
                    if not qc_violations.empty:
                        for idx, row in qc_violations.iterrows():
                            qc_messages.append(
                                f"❌ {row['Event_Number']}: Outcome is 'Unsuccessful' and 'Defensive_Skill' is empty."
                            )
                        # Optional: save to CSV
                        qc_violations.to_csv("qc_violations.csv", index=False)
                    else:
                        qc_messages.append("✅ QC 13: All rows are correct.")

                        
                    st.session_state.processed_df = df  # processed dataframe
                    st.session_state.qc_results = qc_messages  # QC messages
                    st.success("Processing and QC complete!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    reset_state()

    # --- Display QC Results ---
    if st.session_state.qc_results:
        st.header("Quality Check (QC) Results")
        qc_output_string = "\n".join(st.session_state.qc_results)
        st.code(qc_output_string, language='text')

    # --- Download Processed Data ---
    if st.session_state.processed_df is not None:
        st.header("Download Processed Data")
        csv_buffer = io.StringIO()
        st.session_state.processed_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Download Processed CSV",
            data=csv_buffer.getvalue().encode('utf-8'),
            file_name="processed_kabaddi_data.csv",
            mime="text/csv"
        )

        st.subheader("Preview Processed Data")

        rows, cols = st.session_state.processed_df.shape
        st.write(f"**Total rows:** `{rows}` | **Total columns:** `{cols}`")
        
        st.dataframe(st.session_state.processed_df.head(5), use_container_width=True)
    else:
        if not uploaded_file:

            st.info("Upload a CSV file to start processing.")

