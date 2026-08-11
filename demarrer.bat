@echo off
echo ========================================
echo  TickerLab — Demarrage du serveur
echo ========================================
echo.
echo Branche requise : feat/web-configurator
echo URL : http://localhost:8000/landing.html
echo.
echo Appuyez sur Ctrl+C pour arreter.
echo.
python -m web.app
pause
