import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
from datetime import datetime

st.set_page_config(layout="wide", page_title="強制指定店舖轉貨系統", page_icon="📦")


def preprocess_data(df):
    """
    Preprocesses the uploaded data according to the specified rules.
    """
    notes = []
    required_columns = [
        'Article', 'Article Description', 'RP Type', 'Site', 'OM', 'MOQ',
        'SaSa Net Stock', 'Target', 'Pending Received', 'Safety Stock',
        'Last Month Sold Qty', 'MTD Sold Qty'
    ]

    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"錯誤：缺失必需的欄位: {', '.join(missing_cols)}")
        return None, None

    # Add Notes column
    df['Notes'] = ''

    # 1. Force 'Article' to string
    df['Article'] = df['Article'].astype(str)

    # 2. Convert quantity fields to integer, fill invalid values with 0
    quantity_cols = [
        'MOQ', 'SaSa Net Stock', 'Target', 'Pending Received', 'Safety Stock',
        'Last Month Sold Qty', 'MTD Sold Qty'
    ]
    for col in quantity_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        notes.append(f"'{col}' converted to integer, invalid values set to 0.")

    # 3. Correct negative stock and sales to 0
    for col in ['SaSa Net Stock', 'Last Month Sold Qty', 'MTD Sold Qty']:
        negative_mask = df[col] < 0
        if negative_mask.any():
            df.loc[negative_mask, 'Notes'] += f"Corrected negative '{col}' to 0; "
            df.loc[negative_mask, col] = 0
            notes.append(f"Negative values in '{col}' corrected to 0.")

    # 4. Handle sales outliers
    for col in ['Last Month Sold Qty', 'MTD Sold Qty']:
        outlier_mask = df[col] > 100000
        if outlier_mask.any():
            df.loc[outlier_mask, 'Notes'] += f"Capped '{col}' from >100000; "
            df.loc[outlier_mask, col] = 100000
            notes.append(f"Outliers in '{col}' (>100000) capped at 100000.")

    # 5. Fill empty string fields
    string_cols = ['Article Description', 'RP Type', 'Site', 'OM']
    for col in string_cols:
        df[col] = df[col].fillna('')
        notes.append(f"Null values in '{col}' filled with empty string.")

    # Validate RP Type
    invalid_rp_types = df[~df['RP Type'].isin(['ND', 'RF', ''])]
    if not invalid_rp_types.empty:
        st.error(f"錯誤：RP Type 欄位包含無效值: {invalid_rp_types['RP Type'].unique()}")
        st.info("RP Type 只接受 'ND' 或 'RF'。")
        return None, None

    # 6. Record data cleaning log
    df['Notes'] = df['Notes'].str.strip()
    st.session_state.processing_log = notes

    return df, notes


def calculate_conservative_transfers(df):
    """
    Calculates transfer recommendations based on the conservative strategy.
    """
    df['Effective Sale'] = np.where(df['Last Month Sold Qty'] > 0, df['Last Month Sold Qty'], df['MTD Sold Qty'])
    
    # Identify transfer-out candidates
    
    # Priority 1: ND type full transfer
    nd_transfers = df[(df['RP Type'] == 'ND') & (df['SaSa Net Stock'] > 0)].copy()
    nd_transfers['Transfer Qty'] = nd_transfers['SaSa Net Stock']
    nd_transfers['Transfer Type'] = 'ND轉出'

    # Priority 2: RF type excess transfer
    rf_transfers = df[(df['RP Type'] == 'RF') & ((df['SaSa Net Stock'] + df['Pending Received']) > df['Safety Stock'])].copy()
    if not rf_transfers.empty:
        rf_transfers['Max Sale'] = rf_transfers.groupby('Article')['Effective Sale'].transform('max')
        rf_transfers = rf_transfers[rf_transfers['Effective Sale'] < rf_transfers['Max Sale']]
        
        rf_transfers['Base Transferable'] = (rf_transfers['SaSa Net Stock'] + rf_transfers['Pending Received']) - rf_transfers['Safety Stock']
        rf_transfers['Cap Control'] = (rf_transfers['SaSa Net Stock'] + rf_transfers['Pending Received']) * 0.5
        rf_transfers['Transfer Qty'] = np.minimum(rf_transfers['Base Transferable'], rf_transfers['Cap Control']).astype(int)
        rf_transfers['Transfer Qty'] = np.minimum(rf_transfers['Transfer Qty'], rf_transfers['SaSa Net Stock'])
        rf_transfers['Transfer Type'] = 'RF過剩轉出'
        rf_transfers = rf_transfers[rf_transfers['Transfer Qty'] > 0]

    transfer_out_candidates = pd.concat([nd_transfers, rf_transfers]).sort_values(by=['Article', 'RP Type', 'Effective Sale'])

    # Identify receive candidates
    receive_candidates = df[df['Target'] > 0].copy()
    receive_candidates['Receive Qty'] = receive_candidates['Target']
    receive_candidates['Receive Type'] = '指定店舖補貨'
    receive_candidates = receive_candidates.sort_values(by=['Article', 'Target'], ascending=[True, False])

    # Matching logic
    recommendations = []
    
    total_demand_map = receive_candidates.groupby(['Article', 'OM'])['Receive Qty'].sum().to_dict()
    total_transferred_map = {}
    
    for article, group in transfer_out_candidates.groupby('Article'):
        om_group = group.groupby('OM')
        for om, out_group in om_group:
            
            receivers = receive_candidates[(receive_candidates['Article'] == article) & (receive_candidates['OM'] == om)]
            
            if receivers.empty:
                continue

            out_group = out_group.sort_values(by='Effective Sale') # Sort by sales for transfer order
            
            for _, receiver in receivers.iterrows():
                needed = receiver['Receive Qty']
                if needed <= 0:
                    continue
                
                for _, transferer in out_group.iterrows():
                    if transferer['Site'] == receiver['Site']: # Avoid self-transfer
                        continue
                    
                    available = transferer['Transfer Qty']
                    if available <= 0:
                        continue
                        
                    key = (article, om)
                    demand_limit = total_demand_map.get(key, 0)
                    current_transferred = total_transferred_map.get(key, 0)
                    
                    if current_transferred >= demand_limit:
                        continue

                    remaining_allowed = demand_limit - current_transferred
                    transfer_qty = min(needed, available, remaining_allowed)
                    
                    if transfer_qty > 0:
                        recommendations.append({
                            'Article': article,
                            'Article Description': transferer['Article Description'],
                            'From OM': om,
                            'From Site': transferer['Site'],
                            'To Site': receiver['Site'],
                            'Transfer Qty': transfer_qty,
                            'Transfer Type': transferer['Transfer Type'],
                            'Receive Type': receiver['Receive Type']
                        })
                        
                        total_transferred_map[key] = current_transferred + transfer_qty
                        
                        needed -= transfer_qty
                        transferer['Transfer Qty'] -= transfer_qty
                    
                    if needed == 0:
                        break
    
    return pd.DataFrame(recommendations)


def calculate_aggressive_transfers(df):
    """
    Calculates transfer recommendations based on the aggressive strategy.
    """
    df['Effective Sale'] = np.where(df['Last Month Sold Qty'] > 0, df['Last Month Sold Qty'], df['MTD Sold Qty'])

    # Priority 1: ND type full transfer
    nd_transfers = df[(df['RP Type'] == 'ND') & (df['SaSa Net Stock'] > 0)].copy()
    nd_transfers['Transfer Qty'] = nd_transfers['SaSa Net Stock']
    nd_transfers['Transfer Type'] = 'ND轉出'

    # Priority 2: RF type aggressive transfer
    rf_transfers = df[(df['RP Type'] == 'RF') & ((df['SaSa Net Stock'] + df['Pending Received']) > (df['MOQ']))].copy()
    if not rf_transfers.empty:
        rf_transfers['Max Sale'] = rf_transfers.groupby('Article')['Effective Sale'].transform('max')
        rf_transfers = rf_transfers[rf_transfers['Effective Sale'] < rf_transfers['Max Sale']]

        rf_transfers['Base Transferable'] = (rf_transfers['SaSa Net Stock'] + rf_transfers['Pending Received']) - (rf_transfers['MOQ'])
        rf_transfers['Cap Control'] = (rf_transfers['SaSa Net Stock'] + rf_transfers['Pending Received']) * 0.9
        rf_transfers['Transfer Qty'] = np.minimum(rf_transfers['Base Transferable'], rf_transfers['Cap Control']).astype(int)
        rf_transfers['Transfer Qty'] = np.minimum(rf_transfers['Transfer Qty'], rf_transfers['SaSa Net Stock'])
        rf_transfers['Transfer Type'] = 'RF加強轉出'
        rf_transfers = rf_transfers[rf_transfers['Transfer Qty'] > 0]

    transfer_out_candidates = pd.concat([nd_transfers, rf_transfers]).sort_values(by=['Article', 'RP Type', 'Effective Sale'])

    # Identify receive candidates
    receive_candidates = df[df['Target'] > 0].copy()
    receive_candidates['Receive Qty'] = receive_candidates['Target']
    receive_candidates['Receive Type'] = '指定店舖補貨'
    receive_candidates = receive_candidates.sort_values(by=['Article', 'Target'], ascending=[True, False])

    # Matching logic (same as conservative)
    recommendations = []
    
    total_demand_map = receive_candidates.groupby(['Article', 'OM'])['Receive Qty'].sum().to_dict()
    total_transferred_map = {}

    for article, group in transfer_out_candidates.groupby('Article'):
        om_group = group.groupby('OM')
        for om, out_group in om_group:
            receivers = receive_candidates[(receive_candidates['Article'] == article) & (receive_candidates['OM'] == om)]
            if receivers.empty:
                continue

            out_group = out_group.sort_values(by='Effective Sale')
            for _, receiver in receivers.iterrows():
                needed = receiver['Receive Qty']
                if needed <= 0:
                    continue
                
                for _, transferer in out_group.iterrows():
                    if transferer['Site'] == receiver['Site']:
                        continue
                    
                    available = transferer['Transfer Qty']
                    if available <= 0:
                        continue
                        
                    key = (article, om)
                    demand_limit = total_demand_map.get(key, 0)
                    current_transferred = total_transferred_map.get(key, 0)
                    
                    if current_transferred >= demand_limit:
                        continue

                    remaining_allowed = demand_limit - current_transferred
                    transfer_qty = min(needed, available, remaining_allowed)
                    
                    if transfer_qty > 0:
                        recommendations.append({
                            'Article': article,
                            'Article Description': transferer['Article Description'],
                            'From OM': om,
                            'From Site': transferer['Site'],
                            'To Site': receiver['Site'],
                            'Transfer Qty': transfer_qty,
                            'Transfer Type': transferer['Transfer Type'],
                            'Receive Type': receiver['Receive Type']
                        })
                        
                        total_transferred_map[key] = current_transferred + transfer_qty
                        
                        needed -= transfer_qty
                        transferer['Transfer Qty'] -= transfer_qty
                    
                    if needed == 0:
                        break
    
    return pd.DataFrame(recommendations)


def calculate_super_aggressive_transfers(df):
    df['Effective Sale'] = np.where(df['Last Month Sold Qty'] > 0, df['Last Month Sold Qty'], df['MTD Sold Qty'])
    transfer_out_list = []

    # Priority 1: ND Full Transfer
    nd_transfers = df[(df['RP Type'] == 'ND') & (df['SaSa Net Stock'] > 0)].copy()
    nd_transfers['Transfer Out Qty'] = nd_transfers['SaSa Net Stock']
    nd_transfers['Transfer Out Type'] = 'ND轉出'
    nd_transfers['Priority'] = 1
    transfer_out_list.append(nd_transfers)

    # Priority 2: RF Super Enhanced Transfer (ignore MOQ/Safety, allow up to 100%)
    rf_super_aggressive = df[(df['RP Type'] == 'RF') & (df['SaSa Net Stock'] > 0)].copy()

    if not rf_super_aggressive.empty:
        rf_super_aggressive = rf_super_aggressive.sort_values(by=['Article', 'Effective Sale'], ascending=[True, True])
        
        # Calculations
        rf_super_aggressive['Transfer Out Qty'] = rf_super_aggressive['SaSa Net Stock'].astype(int)

        rf_super_aggressive = rf_super_aggressive[rf_super_aggressive['Transfer Out Qty'] > 0]
        rf_super_aggressive['Transfer Out Type'] = 'RF特強轉出'
        rf_super_aggressive['Priority'] = 2
        transfer_out_list.append(rf_super_aggressive)

    if not transfer_out_list:
        return pd.DataFrame()

    transfer_out = pd.concat(transfer_out_list).sort_values(by=['Article', 'Priority'])

    # --- Identify Receive Candidates ---
    receive_in = df[df['Target'] > 0].copy()
    receive_in['Receive In Qty'] = receive_in['Target']
    receive_in['Receive In Type'] = '指定店舖補貨'
    receive_in = receive_in.sort_values(by=['Article', 'Target'], ascending=[True, False])

    # --- Matching Logic ---
    recommendations = []
    transfer_out_grouped = transfer_out.groupby(['Article', 'OM'])
    receive_in_grouped = receive_in.groupby(['Article', 'OM'])

    total_demand_map = receive_in.groupby(['Article', 'OM'])['Receive In Qty'].sum().to_dict()
    total_transferred_map = {}

    for (article, om), out_group in transfer_out_grouped:
        if (article, om) in receive_in_grouped.groups:
            in_group = receive_in_grouped.get_group((article, om))
            
            for _, out_row in out_group.iterrows():
                for _, in_row in in_group.iterrows():
                    if out_row['Site'] != in_row['Site'] and out_row['Transfer Out Qty'] > 0 and in_row['Receive In Qty'] > 0:
                        
                        key = (article, om)
                        demand_limit = total_demand_map.get(key, 0)
                        current_transferred = total_transferred_map.get(key, 0)

                        if current_transferred >= demand_limit:
                            continue

                        remaining_allowed = demand_limit - current_transferred
                        transfer_qty = min(out_row['Transfer Out Qty'], in_row['Receive In Qty'], remaining_allowed)
                        
                        if transfer_qty > 0:
                            recommendations.append({
                                'From Site': out_row['Site'],
                                'To Site': in_row['Site'],
                                'Article': article,
                                'Article Description': out_row['Article Description'],
                                'From OM': om,
                                'Transfer Qty': transfer_qty,
                                'Transfer Type': out_row['Transfer Out Type'],
                                'Receive Type': in_row['Receive In Type']
                            })
                            
                            total_transferred_map[key] = current_transferred + transfer_qty
                            
                            # Update quantities
                            out_row['Transfer Out Qty'] -= transfer_qty
                            in_row['Receive In Qty'] -= transfer_qty

    if not recommendations:
        return pd.DataFrame()
        
    return pd.DataFrame(recommendations)

def display_statistics(recs, original_df):
    st.header("5. 統計分析")

    if recs.empty:
        st.info("沒有調貨建議可供分析。")
        return

    # 1. KPIs
    total_recs = len(recs)
    total_qty = recs['Transfer Qty'].sum()
    unique_articles = recs['Article'].nunique()
    unique_oms = recs['From OM'].nunique()

    kpi_data = {
        "指標": ["總調貨建議行數", "總調貨件數", "涉及SKU數量", "涉及OM數量"],
        "數值": [total_recs, total_qty, unique_articles, unique_oms]
    }
    kpi_df = pd.DataFrame(kpi_data)

    st.subheader("📊 KPI 概覽")
    st.table(kpi_df)

    # 2. Per-Article Statistics
    st.subheader("📦 按SKU統計")
    article_stats = recs.groupby('Article').agg(
        Total_Demand=('Transfer Qty', 'sum'),
        Total_Transferred=('Transfer Qty', 'sum'),
        Transfer_Lines=('Article', 'count')
    ).reset_index()
    article_stats['Fulfillment_Rate'] = (article_stats['Total_Transferred'] / article_stats['Total_Demand'].replace(0, 1)).apply(lambda x: f"{x:.2%}")
    st.dataframe(article_stats)

    # 3. Per-OM Statistics
    st.subheader("🏢 按OM統計")
    om_stats = recs.groupby('From OM').agg(
        Total_Transferred=('Transfer Qty', 'sum'),
        Transfer_Lines=('From OM', 'count'),
        Unique_Articles=('Article', 'nunique')
    ).reset_index()
    st.dataframe(om_stats)

    # 4. Transfer-out type distribution
    st.subheader("🚚 轉出類型分佈")
    
    strategy = st.session_state.get('transfer_strategy', '保守轉貨 (Conservative)')
    
    if strategy == '保守轉貨 (Conservative)':
        transfer_type_summary = recs.groupby('Transfer Type').agg(
            Total_Qty=('Transfer Qty', 'sum'),
            Lines=('Transfer Type', 'count')
        ).reset_index()
        st.dataframe(transfer_type_summary)
    elif strategy == '加強轉貨 (Aggressive)':
        transfer_type_summary = recs.groupby('Transfer Type').agg(
            Total_Qty=('Transfer Qty', 'sum'),
            Lines=('Transfer Type', 'count')
        ).reset_index()
        st.dataframe(transfer_type_summary)
    elif strategy == '超級增強轉貨 (Super Enhanced)':
        transfer_type_summary = recs.groupby('Transfer Type').agg(
            Total_Qty=('Transfer Qty', 'sum'),
            Lines=('Transfer Type', 'count')
        ).reset_index()
        st.dataframe(transfer_type_summary)


    # 5. Receive type results
    st.subheader("🎯 接收類型結果")
    receive_summary = recs.groupby('To Site').agg(
        Demand_Qty=('Transfer Qty', 'sum'),
        Received_Qty=('Transfer Qty', 'sum')
    ).reset_index()
    st.dataframe(receive_summary)
    
    st.session_state['stats'] = {
        "kpi": kpi_df,
        "article_stats": article_stats,
        "om_stats": om_stats,
        "transfer_type_summary": transfer_type_summary,
        "receive_summary": receive_summary
    }

def create_visualizations(recs, original_df, strategy):
    st.header("6. 資料視覺化")

    if recs.empty:
        st.info("沒有資料可供視覺化。")
        return

    # Group data by OM for plotting
    om_agg = recs.groupby('From OM').agg(
        ND_Transfer_Qty=('Transfer Qty', lambda x: x[recs['Transfer Type'] == 'ND轉出'].sum()),
        RF_Surplus_Qty=('Transfer Qty', lambda x: x[recs['Transfer Type'] == 'RF過剩轉出'].sum()),
        RF_Aggressive_Qty=('Transfer Qty', lambda x: x[recs['Transfer Type'] == 'RF加強轉出'].sum()),
        RF_Super_Aggressive_Qty=('Transfer Qty', lambda x: x[recs['Transfer Type'] == 'RF特強轉出'].sum()),
        Demand_Qty=('Transfer Qty', 'sum'),  # Placeholder for total demand
        Actual_Receive_Qty=('Transfer Qty', 'sum')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    
    oms = om_agg['From OM']
    x = np.arange(len(oms))
    width = 0.15

    if strategy == '保守轉貨 (Conservative)':
        ax.bar(x - width, om_agg['ND_Transfer_Qty'], width, label='ND Transfer')
        ax.bar(x, om_agg['RF_Surplus_Qty'], width, label='RF Surplus')
        ax.bar(x + width, om_agg['Demand_Qty'], width, label='Demand')
        ax.bar(x + 2*width, om_agg['Actual_Receive_Qty'], width, label='Actual Received')
    elif strategy == '加強轉貨 (Aggressive)':
        ax.bar(x - 1.5*width, om_agg['ND_Transfer_Qty'], width, label='ND Transfer')
        ax.bar(x - 0.5*width, om_agg['RF_Surplus_Qty'], width, label='RF Surplus')
        ax.bar(x + 0.5*width, om_agg['RF_Aggressive_Qty'], width, label='RF 加強')
        ax.bar(x + 1.5*width, om_agg['Demand_Qty'], width, label='Demand')
        ax.bar(x + 2.5*width, om_agg['Actual_Receive_Qty'], width, label='Actual Received')
    elif strategy == '超級增強轉貨 (Super Enhanced)':
        ax.bar(x - 1.5*width, om_agg['ND_Transfer_Qty'], width, label='ND Transfer')
        ax.bar(x - 0.5*width, om_agg['RF_Surplus_Qty'], width, label='RF Surplus')
        ax.bar(x + 0.5*width, om_agg['RF_Super_Aggressive_Qty'], width, label='RF 特強')
        ax.bar(x + 1.5*width, om_agg['Demand_Qty'], width, label='Demand')
        ax.bar(x + 2.5*width, om_agg['Actual_Receive_Qty'], width, label='Actual Received')

    ax.set_ylabel('調貨件數')
    ax.set_title('轉出與接收分析')
    ax.set_xticks(x)
    ax.set_xticklabels(oms, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

def export_to_excel(recs, stats, original_df):
    st.header("7. 匯出功能")
    if recs.empty:
        st.info("沒有建議可供匯出。")
        return

    # Prepare original data for merging
    from_site_df = original_df[['Site', 'Article', 'SaSa Net Stock', 'Safety Stock', 'MOQ']].rename(columns={
        'Site': 'From Site',
        'SaSa Net Stock': 'Transfer Site Original Stock',
        'Safety Stock': 'Transfer Site Safety Stock',
        'MOQ': 'Transfer Site MOQ'
    })

    to_site_df = original_df[['Site', 'Article', 'Target']].rename(columns={
        'Site': 'To Site',
        'Target': 'Receive Site Target Qty'
    })

    # Merge with recommendations
    export_df = recs.merge(from_site_df, on=['From Site', 'Article'], how='left')
    export_df = export_df.merge(to_site_df, on=['To Site', 'Article'], how='left')


    # Create a new DataFrame for export
    export_df_final = pd.DataFrame()

    # Add columns in the specified order
    export_df_final['Article'] = export_df['Article']
    export_df_final['Product Desc'] = export_df['Article'].astype(str) + " " + export_df['Article Description']
    export_df_final['OM'] = export_df['From OM']
    export_df_final['Transfer Site'] = export_df['From Site']
    export_df_final['Transfer Qty'] = export_df['Transfer Qty']
    
    # Populate with actual data
    export_df_final['Transfer Site Original Stock'] = export_df['Transfer Site Original Stock']
    export_df_final['Transfer Site After Transfer Stock'] = export_df['Transfer Site Original Stock'] - export_df['Transfer Qty']
    export_df_final['Transfer Site Safety Stock'] = export_df['Transfer Site Safety Stock']
    export_df_final['Transfer Site MOQ'] = export_df['Transfer Site MOQ']
    export_df_final['Receive Site'] = export_df['To Site']
    export_df_final['Receive Site Target Qty'] = export_df['Receive Site Target Qty']
    export_df_final['Notes'] = ''

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        export_df_final.to_excel(writer, sheet_name='Transfer Suggestions', index=False)

    filename = f"Transfer_Suggestions_{datetime.now().strftime('%Y%m%d')}.xlsx"
    st.download_button(
        label="📥 下載 Excel 報表",
        data=excel_buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def main():
    st.title("📦 強制指定店舖轉貨系統")
    st.sidebar.header("系統資訊")
    st.sidebar.info("""
    **版本：v1.0**
    **開發者:Ricky**

    **核心功能：**
    - ✅ ND/RF類型智慧識別
    - ✅ 優先順序轉貨
    - ✅ 統計分析和圖表
    - ✅ Excel格式匯出
    """)

    st.header("1. 資料上傳")
    uploaded_file = st.file_uploader(
        "請上傳Excel檔",
        type=['xlsx', 'xls'],
        help="請確保檔案包含 'Article', 'Article Description', 'RP Type', 'Site', 'OM', 'MOQ', 'SaSa Net Stock', 'Target', 'Pending Received', 'Safety Stock', 'Last Month Sold Qty', 'MTD Sold Qty' 等欄位。"
    )

    if uploaded_file:
        try:
            progress_bar = st.progress(0)
            with st.spinner('正在讀取與處理檔案...'):
                progress_bar.progress(25, text="正在讀取 Excel 檔案...")
                df = pd.read_excel(uploaded_file)
                progress_bar.progress(50, text="檔案讀取完成，正在進行資料預處理...")

                if df.empty:
                    st.error("錯誤：上傳的檔案為空，請檢查檔案內容。")
                    return

                processed_df, notes = preprocess_data(df)
                progress_bar.progress(100, text="資料預處理完成！")

                if processed_df is not None:
                    st.session_state.df = processed_df
                    st.success("檔案上傳與預處理成功！")

                    st.header("2. 資料預覽")
                    st.metric("總記錄數", f"{len(processed_df):,}")
                    st.dataframe(processed_df.head())

                    with st.expander("查看資料清理紀錄"):
                        for note in notes:
                            st.info(note)

                    st.header("3. 分析選項")
                    transfer_strategy = st.radio(
                        "請選擇調貨模式：",
                        ('保守轉貨 (Conservative)', '加強轉貨 (Aggressive)', '超級增強轉貨 (Super Enhanced)'),
                        help="保守模式優先處理 ND 與過剩庫存；加強模式更積極地由低銷量店舖轉出；超級增強模式最為積極。"
                    )

                    if st.button("🚀 開始分析"):
                        st.session_state.transfer_strategy = transfer_strategy
                        
                        with st.spinner('正在產生調貨建議...'):
                            if 'df' in st.session_state:
                                recommendations = None
                                if transfer_strategy == '保守轉貨 (Conservative)':
                                    recommendations = calculate_conservative_transfers(st.session_state.df)
                                elif transfer_strategy == '加強轉貨 (Aggressive)':
                                    recommendations = calculate_aggressive_transfers(st.session_state.df)
                                elif transfer_strategy == '超級增強轉貨 (Super Enhanced)':
                                    recommendations = calculate_super_aggressive_transfers(st.session_state.df)
                                
                                if recommendations is not None:
                                    if not recommendations.empty:
                                        st.header("4. 結果展示")
                                        st.success(f"成功產生 {len(recommendations)} 筆調貨建議！")
                                        st.dataframe(recommendations)
                                        
                                        with st.spinner('正在產生統計與圖表...'):
                                            display_statistics(recommendations, st.session_state.df)
                                            create_visualizations(recommendations, st.session_state.df, st.session_state.transfer_strategy)
                                            export_to_excel(recommendations, st.session_state.get('stats', {}), df)
                                    else:
                                        st.info("依所選模式，未產生任何調貨建議。")
                                else:
                                    st.error("請先上傳檔案。")

        except Exception as e:
            st.error(f"處理檔案時發生錯誤: {e}")
            st.info("請檢查檔案格式是否正確，或欄位是否符合要求。可能為記憶體不足或檔案已損壞。")


if __name__ == "__main__":
    main()
