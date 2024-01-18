@echo off

rem Check the number of arguments
if "%1"=="" (
    echo Usage: %0 ^<nombre_de_joueurs^>
    exit /B 1
)

rem Retrieve the number of players from the first argument
set "nb_joueurs=%1"

rem Directory of the game
set "jeu_dossier=C:\Users\nayel\Documents\Git\Robot\scripts"

rem Launch the server in the first command prompt window
start "Serveur" cmd /K "cd /D %jeu_dossier% && python server.py -nb %nb_joueurs%"

rem Wait for a moment to allow the server to start
timeout /T 2 /NOBREAK >nul

rem Launch the agents in subsequent command prompt windows
for /L %%i in (1, 1, %nb_joueurs%) do (
    start "Agent %%i" cmd /K "cd /D %jeu_dossier% && python agent.py"
)

exit /B 0
