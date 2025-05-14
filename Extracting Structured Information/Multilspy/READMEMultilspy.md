# Analyse statique du projet Astropy avec multilspy

Ce projet utilise la bibliothèque Python **multilspy** pour effectuer une analyse statique du code source du dépôt [astropy-main](https://github.com/astropy/astropy). L'objectif est d'extraire des métadonnées pertinentes pour une meilleure compréhension du code par des modèles de langage.

## Prérequis

* Python ≥ 3.10
* Installation de multilspy :

  ```bash
  pip install multilspy
  ```

## Fonctionnalités

* **Extraction des symboles** : identifie les fonctions, classes et variables globales dans chaque fichier Python.
* **Définitions et références** : localise les définitions et les usages de ces symboles dans le projet.
* **Analyse AST (Abstract Syntax Tree)** : extrait les classes, fonctions et imports via le module `ast` de Python.
* **Structure du projet** : cartographie les répertoires et fichiers du projet.

## Utilisation

Le script `extract_metadata.py` parcourt récursivement les fichiers `.py` du projet, extrait les métadonnées et les enregistre dans un fichier JSON :

```bash
python extract_metadata.py
```

## Structure du fichier JSON généré

Le fichier `project_metadata_astropy.json` contient :

* `project_structure` : liste des répertoires et fichiers.
* `files` : métadonnées extraites pour chaque fichier Python, incluant :

  * `file` : chemin relatif du fichier.
  * `symbols` : symboles extraits via LSP.
  * `references` : références des symboles.
  * `definitions` : définitions des symboles.
  * `ast` : métadonnées issues de l'analyse AST.

## Exemple de contenu JSON

```json
{
  "project_structure": [
    {"directory": "astropy", "files": ["core.py", "utils.py"]},
    {"directory": "docs", "files": ["index.rst"]}
  ],
  "files": [
    {
      "file": "astropy/core.py",
      "symbols": [{"name": "function1", "kind": "function", "location": {"line": 10, "character": 4}}],
      "references": [...],
      "definitions": [...],
      "ast": {"functions": ["function1"], "classes": ["ClassA"], "imports": ["numpy"]}
    }
  ]
}
```

## À propos de multilspy

[multilspy](https://github.com/microsoft/multilspy) est une bibliothèque Python développée par Microsoft pour interagir avec des serveurs de langage via le protocole LSP. Elle facilite l'extraction d'informations statiques sur le code, telles que les symboles, les définitions et les références, en offrant une interface unifiée pour différents langages de programmation.
