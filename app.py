import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

from io import BytesIO
from datetime import datetime

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


def display_statistics(recs_df, initial_df):
    st.header("5. 統計分析")

    if recs_df.empty:
        st.info("沒有建議可供分析。")
        return

    # 1. Basic KPIs
    total_recs = len(recs_df)
    total_qty = recs_df['Transfer Qty'].sum()
    involved_articles = recs_df['Article'].nunique()
    involved_oms = recs_df['OM'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("總調貨建議數量", f"{total_recs:,}")
    col2.metric("總調貨件數", f"{total_qty:,.0f}")
    col3.metric("涉及產品數量", f"{involved_articles:,}")
    col4.metric("涉及OM數量", f"{involved_oms:,}")

    st.markdown("---")

    # 2. Per Article Stats
    article_stats = recs_df.groupby('Article').agg(
        Total_Transfer_Qty=('Transfer Qty', 'sum'),
        Recommendation_Lines=('Article', 'size'),
        Involved_OMs=('OM', 'nunique')
    ).reset_index()
    article_stats = article_stats.sort_values(by="Total_Transfer_Qty", ascending=False)

    # 3. Per OM Stats
    om_stats = recs_df.groupby('OM').agg(
        Total_Transfer_Qty=('Transfer Qty', 'sum'),
        Recommendation_Lines=('OM', 'size'),
        Involved_Articles=('Article', 'nunique')
    ).reset_index()
    om_stats = om_stats.sort_values(by="Total_Transfer_Qty", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("按產品統計")
        st.dataframe(article_stats.style.format({"Total_Transfer_Qty": "{:,.0f}"}))
    with col2:
        st.subheader("按OM統計")
        st.dataframe(om_stats.style.format({"Total_Transfer_Qty": "{:,.0f}"}))

    st.markdown("---")

    # 4. Transfer-out Type Distribution
    transfer_type_stats = recs_df['Notes'].apply(lambda x: x.split(' -> ')[0]).value_counts().reset_index()
    transfer_type_stats.columns = ['Transfer Type', 'Recommendation_Lines']
    
    transfer_qty_by_type = recs_df.groupby(recs_df['Notes'].apply(lambda x: x.split(' -> ')[0]))['Transfer Qty'].sum().reset_index()
    transfer_qty_by_type.columns = ['Transfer Type', 'Total_Transfer_Qty']

    transfer_type_summary = pd.merge(transfer_type_stats, transfer_qty_by_type, on='Transfer Type')


    # 5. Receive Type Results
    target_demand = initial_df[initial_df['Target'] > 0].groupby(['Article', 'OM'])['Target'].sum().reset_index()
    target_demand.rename(columns={'Target': 'Total_Target_Qty'}, inplace=True)

    actual_received = recs_df.groupby(['Article', 'OM'])['Transfer Qty'].sum().reset_index()
    actual_received.rename(columns={'Transfer Qty': 'Actual_Received_Qty'}, inplace=True)

    receive_summary = pd.merge(target_demand, actual_received, on=['Article', 'OM'], how='left').fillna(0)
    receive_summary['Fulfillment_Rate'] = (receive_summary['Actual_Received_Qty'] / receive_summary['Total_Target_Qty']).fillna(0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("轉出類型分佈")
        st.dataframe(transfer_type_summary.style.format({"Total_Transfer_Qty": "{:,.0f}"}))
    with col2:
        st.subheader("接收類型結果 (指定需求補貨)")
        st.dataframe(receive_summary.style.format({
            "Total_Target_Qty": "{:,.0f}",
            "Actual_Received_Qty": "{:,.0f}",
            "Fulfillment_Rate": "{:.2%}"
        }))
    
    st.session_state.stats = {
        "kpi": {
            "total_recs": total_recs,
            "total_qty": total_qty,
            "involved_articles": involved_articles,
            "involved_oms": involved_oms,
        },
        "article_stats": article_stats,
        "om_stats": om_stats,
        "transfer_type_summary": transfer_type_summary,
        "receive_summary": receive_summary
    }


def create_visualizations(recs_df, initial_df, strategy):
    st.header("視覺化分析")

    if recs_df.empty:
        st.info("沒有建議可供視覺化。")
        return

    # Prepare data for plotting
    om_list = sorted(initial_df['OM'].unique())
    
    # Calculate ND and RF transfer quantities
    nd_qty = recs_df[recs_df['Notes'].str.contains('ND')].groupby('OM')['Transfer Qty'].sum()
    rf_conservative_qty = recs_df[recs_df['Notes'].str.contains('RF過剩轉出')].groupby('OM')['Transfer Qty'].sum()
    rf_aggressive_qty = recs_df[recs_df['Notes'].str.contains('RF加強轉出')].groupby('OM')['Transfer Qty'].sum()
    
    # Calculate demand and actual received quantities
    demand_qty = initial_df[initial_df['Target'] > 0].groupby('OM')['Target'].sum()
    actual_received_qty = recs_df.groupby('OM')['Transfer Qty'].sum()

    # Reindex to ensure all OMs are present
    nd_qty = nd_qty.reindex(om_list, fill_value=0)
    rf_conservative_qty = rf_conservative_qty.reindex(om_list, fill_value=0)
    rf_aggressive_qty = rf_aggressive_qty.reindex(om_list, fill_value=0)
    demand_qty = demand_qty.reindex(om_list, fill_value=0)
    actual_received_qty = actual_received_qty.reindex(om_list, fill_value=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.2
    index = np.arange(len(om_list))

    if strategy == 'A: 保守轉貨':
        ax.bar(index - bar_width, nd_qty, bar_width, label='ND Transfer Qty')
        ax.bar(index, rf_conservative_qty, bar_width, label='RF Excess Transfer Qty')
        ax.bar(index + bar_width, demand_qty, bar_width, label='Demand Receive Qty')
        ax.bar(index + 2 * bar_width, actual_received_qty, bar_width, label='Actual Receive Qty')
        title = 'Transfer Receive Analysis (Conservative)'
    else: # 'B: 加強轉貨'
        ax.bar(index - 1.5 * bar_width, nd_qty, bar_width, label='ND Transfer Qty')
        ax.bar(index - 0.5 * bar_width, rf_aggressive_qty, bar_width, label='RF Aggressive Transfer Qty')
        ax.bar(index + 0.5 * bar_width, demand_qty, bar_width, label='Demand Receive Qty')
        ax.bar(index + 1.5 * bar_width, actual_received_qty, bar_width, label='Actual Receive Qty')
        title = 'Transfer Receive Analysis (Aggressive)'

    ax.set_xlabel('OM')
    ax.set_ylabel('Quantity')
    ax.set_title(title)
    ax.set_xticks(index)
    ax.set_xticklabels(om_list, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)


def export_to_excel(recs_df, stats):
    st.header("6. 匯出功能")
    if recs_df.empty:
        st.info("沒有建議可供匯出。")
        return

    # Sheet 1: Recommendations
    recs_to_export = recs_df[[
        'Article', 'Product Desc', 'OM', 'Transfer Site', 'Transfer Qty',
        'Transfer Site Original Stock', 'Transfer Site After Transfer Stock',
        'Transfer Site Safety Stock', 'Transfer Site MOQ', 'Receive Site',
        'Receive Site Target Qty', 'Notes'
    ]].copy()

    # Sheet 2: Statistics Summary
    kpi_df = pd.DataFrame([stats['kpi']])
    
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        recs_to_export.to_excel(writer, sheet_name='調貨建議', index=False)
        
        start_row = 0
        
        worksheet = writer.sheets['調貨建議']
        # This is a new sheet, so we need to get it by name
        # writer.book.create_sheet('統計摘要')
        # worksheet_stats = writer.sheets['統計摘要']
        
        # KPI
        kpi_df.to_excel(writer, sheet_name='統計摘要', startrow=start_row, index=False)
        worksheet_stats = writer.sheets['統計摘要']
        worksheet_stats.cell(row=start_row + 1, column=1, value="KPI概覽")
        start_row += len(kpi_df) + 3

        # Article Stats
        stats['article_stats'].to_excel(writer, sheet_name='統計摘要', startrow=start_row, index=False)
        worksheet_stats.cell(row=start_row + 1, column=1, value="按Article統計")
        start_row += len(stats['article_stats']) + 3

        # OM Stats
        stats['om_stats'].to_excel(writer, sheet_name='統計摘要', startrow=start_row, index=False)
        worksheet_stats.cell(row=start_row + 1, column=1, value="按OM統計")
        start_row += len(stats['om_stats']) + 3

        # Transfer Type Stats
        stats['transfer_type_summary'].to_excel(writer, sheet_name='統計摘要', startrow=start_row, index=False)
        worksheet_stats.cell(row=start_row + 1, column=1, value="轉出類型分佈")
        start_row += len(stats['transfer_type_summary']) + 3

        # Receive Type Stats
        stats['receive_summary'].to_excel(writer, sheet_name='統計摘要', startrow=start_row, index=False)
        worksheet_stats.cell(row=start_row + 1, column=1, value="接收類型分佈")

    filename = f"強制轉貨建議_{datetime.now().strftime('%Y%m%d')}.xlsx"
    st.download_button(
        label="📥 下載Excel報告",
        data=excel_buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


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
            progress_bar = st.progress(0)
            with st.spinner('正在讀取和處理文件...'):
                progress_bar.progress(25, text="正在讀取Excel文件...")
                df = pd.read_excel(uploaded_file)
                progress_bar.progress(50, text="文件讀取完畢，正在進行數據預處理...")

                if df.empty:
                    st.error("錯誤：上傳的文件為空，請檢查文件內容。")
                    return

                processed_df, notes = preprocess_data(df)
                progress_bar.progress(100, text="數據預處理完成！")

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
                                    
                                    with st.spinner('正在生成統計數據和圖表...'):
                                        display_statistics(recommendations, st.session_state.df)
                                        create_visualizations(recommendations, st.session_state.df, st.session_state.transfer_strategy)
                                        export_to_excel(recommendations, st.session_state.stats)
                                else:
                                    st.info("沒有找到可行的調貨建議。")
                            else:
                                st.error("請先上傳文件。")

        except Exception as e:
            st.error(f"處理文件時發生錯誤: {e}")
            st.info("請檢查文件格式是否正確，或欄位是否符合要求。可能是記憶體不足或文件已損壞。")


if __name__ == "__main__":
    main()