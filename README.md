# 強制指定店鋪轉貨系統

本系統以 Streamlit 構建，依據店舖庫存、在途、補貨目標與銷售數據，生成可執行的店舖間轉貨建議，並提供統計與匯出功能。

## 核心功能

- 智能調貨建議：支援 `ND` 與 `RF` 類型，依三種模式輸出建議
- 視覺化與統計：KPI、SKU/OM 維度統計、轉出類型分佈等
- Excel 匯出：輸出包含轉出/接收站點與轉後庫存等欄位
- 事件驅動：使用者按鈕觸發一次性計算，非即時推流

## 快速開始

- 安裝依賴：`pip install -r requirements.txt`
- 啟動：`streamlit run app.py`
- 上傳 Excel，選擇轉移模式，點擊「開始分析」，查看結果並下載報表

## 數據欄位要求

- 必填欄位：`Article`, `Article Description`, `RP Type`, `Site`, `OM`, `MOQ`, `SaSa Net Stock`, `Target`, `Pending Received`, `Safety Stock`, `Last Month Sold Qty`, `MTD Sold Qty`
- 預處理規則：
  - 數量欄位轉整數，無效值置 0；負值更正為 0；極端銷售值上限 100000
  - `RP Type` 僅接受 `ND`、`RF` 或空值

## 計算定義

- 可用庫存 = `SaSa Net Stock + Pending Received`
- 有效銷售量 = 若 `Last Month Sold Qty > 0` 用之，否則用 `MTD Sold Qty`
- 轉移量不超過實際庫存 `SaSa Net Stock`
- 配對規則：同一 `Article` 與同一 `OM` 之間配對，且 `From Site != To Site`

## 轉移模式邏輯

### Mode A: 保守轉移模式（Conservative）

- ND（Priority 1）
  - 條件：`SaSa Net Stock > 0`
  - 動作：全量轉出
- RF（Priority 2）
  - 條件：`(SaSa Net Stock + Pending Received) > Safety Stock` 且 `有效銷售量 < 該 Article 的最高有效銷售量`
  - 轉移量：`min(可用庫存 - Safety Stock, 可用庫存 * 0.5)`，且不超過 `SaSa Net Stock`
  - 排序：依有效銷售量升序（低銷店優先）

### Mode B: 增強轉移模式（Enhanced）

- ND（Priority 1）
  - 條件：`SaSa Net Stock > 0`
  - 動作：全量轉出
- RF（Priority 2）
  - 條件更新：`(SaSa Net Stock + Pending Received) > MOQ` 且 `有效銷售量 < 該 Article 的最高有效銷售量`
  - 轉移量更新：`min(可用庫存 - MOQ, 可用庫存 * 0.9)`，且不超過 `SaSa Net Stock`
  - 排序：依有效銷售量升序（低銷店優先）

### Mode C: 超級增強轉移模式（Super Enhanced）

- ND（Priority 1）
  - 條件：`SaSa Net Stock > 0`
  - 動作：全量轉出
- RF（Priority 2）
  - 規則更新：忽視最小庫存要求與安全庫存門檻
  - 條件：`SaSa Net Stock > 0`
  - 轉移量：`SaSa Net Stock`（允許 100%）
  - 排序：依有效銷售量升序（過去銷售多者排後）

## 匹配與上限控制

- 接收方以 `Target` 為需求量；同 `Article`、同 `OM` 的總轉出量不超過該組合的需求總和
- 每筆建議包含：`Article`, `Article Description`, `From OM`, `From Site`, `To Site`, `Transfer Qty`, `Transfer Type`, `Receive Type`

## 統計與視覺化

- KPI：建議筆數、總件數、SKU/OM 數量
- 分佈：轉出類型分佈、各 OM 轉出量、需求與實收量
- 視覺化：根據模式顯示 ND/RF 各類別長條圖

## 匯出報表

- 匯出欄位包含：`Article`, `Product Desc`, `OM`, `Transfer Site`, `Transfer Qty`, `Transfer Site Original Stock`, `Transfer Site After Transfer Stock`, `Transfer Site Safety Stock`, `Transfer Site MOQ`, `Receive Site`, `Receive Site Target Qty`, `Notes`
- 下載檔名：`Transfer_Suggestions_YYYYMMDD.xlsx`

## 限制與注意

- 本系統為使用者事件觸發的批次計算，非即時 Pipeline 或背景排程
- 不會跨 `OM` 或相同 `Site` 進行匹配；任何轉移量均不超過轉出站的現庫存
- 預處理會更正負值與極端值；欄位缺失將提示錯誤

## 版本

- v1.0（最新邏輯已實作）：
  - Mode B RF 門檻改為 `可用庫存 > MOQ`，上限改為 90%
  - Mode C RF 允許 100% 轉出，僅需 `SaSa Net Stock > 0`

## 授權與聯絡

- 內部使用方案；如需擴充或調整策略，請聯絡維護者
