# 影像處理與人臉分析工具（Streamlit）

四大功能：影像載入與中繼資訊 ／ RGB・YUV 通道檢視與同步直方圖 ／ 自訂卷積核濾波與 PSNR ／ 人臉偵測。

## 本機執行
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 執行測試
```bash
pip install -r requirements-dev.txt
pytest -q
```

## 部署到 Streamlit Community Cloud
1. 將本資料夾推到 GitHub（公開儲存庫）。
2. 到 https://share.streamlit.io 選擇該儲存庫，Main file path 填 `app.py`，Python 版本選 3.11。
3. 部署完成後，將網址設為 Public 並填入報告第二章。

## 部署到 Hugging Face Spaces
建立 Space 時 SDK 選 Streamlit，上傳本資料夾所有檔案；`README.md` 最上方需加入：
```yaml
---
title: Image Processing Tool
sdk: streamlit
app_file: app.py
python_version: "3.11"
---
```

## 檔案結構
| 檔案 | 說明 |
|---|---|
| `app.py` | Streamlit 介面 |
| `imgproc.py` | 核心演算法（載入、YUV、直方圖、卷積、PSNR、人臉偵測），與 UI 解耦 |
| `tests/test_imgproc.py` | 單元測試（與 scipy / OpenCV / scikit-image 交叉驗證） |
| `assets/` | 範例影像與 LBP cascade 備援模型 |
