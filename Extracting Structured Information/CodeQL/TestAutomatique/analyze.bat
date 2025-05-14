@echo off
REM Vérifie si l'utilisateur a fourni un URL GitHub
IF "%~1"=="" (
    echo Usage: analyze.bat [GitHub Repo URL]
    exit /b 1
)

REM Définir des variables
SET REPO_URL=%~1
SET REPO_NAME=astropy-main
SET DB_NAME=%REPO_NAME%-db
SET OUTPUT_SARIF=resultats_astropy.sarif
SET OUTPUT_JSON=resultats_astropy.json

REM Cloner le dépôt GitHub
echo Clonage du dépôt GitHub...
git clone %REPO_URL% %REPO_NAME%
IF ERRORLEVEL 1 (
    echo Erreur lors du clonage du dépôt.
    exit /b 1
)

REM Créer la base de données CodeQL
echo Création de la base de données CodeQL...
codeql database create %DB_NAME% --language=python --source-root=%REPO_NAME% --overwrite
IF ERRORLEVEL 1 (
    echo Erreur lors de la création de la base de données.
    exit /b 1
)

REM Analyser la base de données avec CodeQL
echo Analyse de la base de données CodeQL...
codeql database analyze %DB_NAME% --format=sarif-latest --output=%OUTPUT_SARIF%
IF ERRORLEVEL 1 (
    echo Erreur lors de l'analyse CodeQL.
    exit /b 1
)

REM Convertir SARIF en JSON avec jq
echo Conversion SARIF vers JSON...
jq "." %OUTPUT_SARIF% > %OUTPUT_JSON%
IF ERRORLEVEL 1 (
    echo Erreur lors de la conversion SARIF -> JSON.
    exit /b 1
)

echo Analyse terminée avec succès.
echo Résultats JSON : %OUTPUT_JSON%
exit /b 0
