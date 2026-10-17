@echo off
chcp 65001 > nul
title VoxSort AI - Phan Loai Video Tu Dong
color 0B

echo ================================================================
echo           VOXSORT AI - PHAN LOAI VIDEO UNBOX ^& REVIEW
echo        Su dung Tri Tue Nhan Tao Faster-Whisper (OpenAI)
echo ================================================================
echo.
echo Dang khoi dong ung dung giao dien...
echo.

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LOI] Khong the chay ung dung. Vui long kiem tra lai moi truong Python!
    pause
)
