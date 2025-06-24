#!/bin/bash

set -e  # Quit on error

# Vérification des arguments
if [ $# -ne 2 ]; then
    echo "Erreur: Arguments manquants."
    echo "Usage: $0 <url-du-dépôt> <hash-du-commit>"
    exit 1
fi

REPO_URL="$1"
COMMIT_HASH="$2"
WORKSPACE_DIR="/fast/scip_workspace"   
REPO_NAME=$(basename "$REPO_URL" .git)
REPO_DIR="$WORKSPACE_DIR/$REPO_NAME"
SCIP_DIR="$WORKSPACE_DIR/scip"

echo "Configuration de l'espace de travail dans $WORKSPACE_DIR..."
mkdir -p "$WORKSPACE_DIR"

echo "Création de l'environnement virtuel..."
python3 -m venv "$WORKSPACE_DIR/.venv"
source "$WORKSPACE_DIR/.venv/bin/activate"

echo "Installation de scip-python via npm..."
scip-python --version

echo "Clonage et compilation de scip..."
# Remove existing scip directory if it exists
if [ -d "$SCIP_DIR" ]; then
    echo "Suppression de l'ancien répertoire scip..."
    rm -rf "$SCIP_DIR"
fi
git clone https://github.com/sourcegraph/scip.git --depth=1 "$SCIP_DIR"
cd "$SCIP_DIR"
go build ./cmd/scip
cd - >/dev/null

echo "Clonage du dépôt cible sans checkout..."
# Remove existing repository directory if it exists
if [ -d "$REPO_DIR" ]; then
    echo "Suppression de l'ancien répertoire du projet..."
    rm -rf "$REPO_DIR"
fi
git clone --no-checkout "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
echo "Checkout vers le commit $COMMIT_HASH"
git checkout "$COMMIT_HASH"

echo "Installation des dépendances Python..."
pip install pytest

echo "Indexation du projet avec scip-python..."
NODE_OPTIONS="--max-old-space-size=8192" scip-python index . --output index.scip

echo "Génération du fichier JSON brut..."
"$SCIP_DIR/scip" print --json index.scip > raw_snapshot.json

echo "Formatage du JSON..."
python3 - <<EOF
import json
with open("raw_snapshot.json", "r") as f:
    data = json.loads(f.read())
with open("formatted_snapshot.json", "w") as f:
    json.dump(data, f, indent=4)
print("Fichier JSON formaté : $REPO_DIR/formatted_snapshot.json")
EOF

echo "------------------------------------------------"
echo "Index SCIP généré avec succès!"
echo "Fichiers disponibles dans: $REPO_DIR"
echo " - Fichier SCIP brut: index.scip"
echo " - Sortie JSON brute: raw_snapshot.json"
echo " - Sortie JSON formatée: formatted_snapshot.json"