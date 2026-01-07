@echo off
chcp 65001
echo ============================================
echo   時系列予測ダッシュボード 起動中...
echo ============================================
echo.
echo ブラウザで http://localhost:8501 を開きます
echo 終了するには Ctrl+C を押してください
echo.
cd /d "%~dp0"
streamlit run forecast_dashboard.py --server.port 8501
pause
