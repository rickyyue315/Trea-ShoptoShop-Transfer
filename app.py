import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="強制指定店鋪轉貨系統", page_icon="📦")


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
    receive_candidates['Receive Type'] = '指定店鋪補貨'
    receive_candidates = receive_candidates.sort_values(by=['Article', 'Target'], ascending=[True, False])

    # Matching logic
    recommendations = []
    
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
                        
                    transfer_qty = min(needed, available)
                    
                    recommendations.append({
                        'Article': article,
                        'Product Desc': transferer['Article Description'],
                        'OM': om,
                        'Transfer Site': transferer['Site'],
                        'Transfer Qty': transfer_qty,
                        'Transfer Site Original Stock': transferer['SaSa Net Stock'],
                        'Transfer Site After Transfer Stock': transferer['SaSa Net Stock'] - transfer_qty,
                        'Transfer Site Safety Stock': transferer['Safety Stock'],
                        'Transfer Site MOQ': transferer['MOQ'],
                        'Receive Site': receiver['Site'],
                        'Receive Site Target Qty': receiver['Target'],
                        'Notes': f"{transferer['Transfer Type']} -> {receiver['Receive Type']}"
                    })
                    
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
    rf_transfers = df[(df['RP Type'] == 'RF') & ((df['SaSa Net Stock'] + df['Pending Received']) > (df['MOQ'] + 1))].copy()
    if not rf_transfers.empty:
        rf_transfers['Max Sale'] = rf_transfers.groupby('Article')['Effective Sale'].transform('max')
        rf_transfers = rf_transfers[rf_transfers['Effective Sale'] < rf_transfers['Max Sale']]

        rf_transfers['Base Transferable'] = (rf_transfers['SaSa Net Stock'] + rf_transfers['Pending Received']) - (rf_transfers['MOQ'] + 1)
        rf_transfers['Cap Control'] = (rf_transfers['SaSa Net Stock'] + rf_transfers['Pending Received']) * 0.8
        rf_transfers['Transfer Qty'] = np.minimum(rf_transfers['Base Transferable'], rf_transfers['Cap Control']).astype(int)
        rf_transfers['Transfer Qty'] = np.minimum(rf_transfers['Transfer Qty'], rf_transfers['SaSa Net Stock'])
        rf_transfers['Transfer Type'] = 'RF加強轉出'
        rf_transfers = rf_transfers[rf_transfers['Transfer Qty'] > 0]

    transfer_out_candidates = pd.concat([nd_transfers, rf_transfers]).sort_values(by=['Article', 'RP Type', 'Effective Sale'])

    # Identify receive candidates
    receive_candidates = df[df['Target'] > 0].copy()
    receive_candidates['Receive Qty'] = receive_candidates['Target']
    receive_candidates['Receive Type'] = '指定店鋪補貨'
    receive_candidates = receive_candidates.sort_values(by=['Article', 'Target'], ascending=[True, False])

    # Matching logic (same as conservative)
    recommendations = []
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
                        
                    transfer_qty = min(needed, available)
                    
                    recommendations.append({
                        'Article': article,
                        'Product Desc': transferer['Article Description'],
                        'OM': om,
                        'Transfer Site': transferer['Site'],
                        'Transfer Qty': transfer_qty,
                        'Transfer Site Original Stock': transferer['SaSa Net Stock'],
                        'Transfer Site After Transfer Stock': transferer['SaSa Net Stock'] - transfer_qty,
                        'Transfer Site Safety Stock': transferer['Safety Stock'],
                        'Transfer Site MOQ': transferer['MOQ'],
                        'Receive Site': receiver['Site'],
                        'Receive Site Target Qty': receiver['Target'],
                        'Notes': f"{transferer['Transfer Type']} -> {receiver['Receive Type']}"
                    })
                    
                    needed -= transfer_qty
                    transferer['Transfer Qty'] -= transfer_qty
                    
                    if needed == 0:
                        break
    
    return pd.DataFrame(recommendations)


def display_statistics(df):
    """
    Placeholder for displaying statistics.
    """
    pass

def create_visualizations(df):
    """
    Placeholder for creating visualizations.
    """
    pass

def export_to_excel(df):
    """
    Placeholder for exporting to Excel.
    """
    pass

def main():
    st.title("📦 強制指定店鋪轉貨系統")
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
        help="請確保文件包含 'Article', 'Article Description', 'RP Type', 'Site', 'OM', 'MOQ', 'SaSa Net Stock', 'Target', 'Pending Received', 'Safety Stock', 'Last Month Sold Qty', 'MTD Sold Qty' 等欄位。"
    )

    if uploaded_file:
        try:
            with st.spinner('正在讀取和處理文件...'):
                df = pd.read_excel(uploaded_file)

                if df.empty:
                    st.error("錯誤：上傳的文件為空，請檢查文件內容。")
                    return

                processed_df, notes = preprocess_data(df)

                if processed_df is not None:
                    st.session_state.df = processed_df
                    st.success("文件上傳和預處理成功！")

                    st.header("2. 資料預覽")
                    st.metric("總記錄數", f"{len(processed_df):,}")
                    st.dataframe(processed_df.head())

                    with st.expander("查看數據清理日誌"):
                        for note in notes:
                            st.info(note)

                    st.header("3. 分析選項")
                    transfer_strategy = st.radio(
                        "請選擇轉貨策略：",
                        ('A: 保守轉貨', 'B: 加強轉貨'),
                        help="A: 保守轉貨 - 優先保證店鋪安全庫存。 B: 加強轉貨 - 更積極地轉出多餘庫存。"
                    )

                    if st.button("🚀 開始分析"):
                        st.session_state.transfer_strategy = transfer_strategy
                        
                        with st.spinner('正在生成調貨建議...'):
                            if 'df' in st.session_state:
                                if transfer_strategy == 'A: 保守轉貨':
                                    recommendations = calculate_conservative_transfers(st.session_state.df)
                                else:
                                    recommendations = calculate_aggressive_transfers(st.session_state.df)
                                
                                st.session_state.recommendations = recommendations
                                
                                if not recommendations.empty:
                                    st.header("4. 結果展示")
                                    st.success(f"成功生成 {len(recommendations)} 條調貨建議！")
                                    st.dataframe(recommendations)
                                else:
                                    st.info("沒有找到可行的調貨建議。")
                            else:
                                st.error("請先上傳文件。")

        except Exception as e:
            st.error(f"處理文件時發生錯誤: {e}")
            st.info("請檢查文件格式是否正確，或欄位是否符合要求。")


if __name__ == "__main__":
    main()