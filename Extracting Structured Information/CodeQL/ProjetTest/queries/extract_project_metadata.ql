/**
 * @name Extraire les métadonnées du projet Python
 * @id custom/extract-python-metadata
 * @description
 *   Cette requête extrait des informations sur la structure du projet Python, 
 *   incluant les fichiers, classes, fonctions et méthodes. L’objectif est 
 *   de fournir un ensemble complet de métadonnées pour qu’un modèle de langage 
 *   spécialisé dans le code (LLM) puisse comprendre le projet dans son ensemble.
 *
 *   Basé sur les étapes présentées dans la documentation "Cross the River" de CodeQL,
 *   les étapes clés sont :
 *     1. Créer un fichier de requête avec un en-tête documenté (@name, @id, @description, etc.).
 *     2. Importer les bibliothèques pertinentes (ici, la librairie Python de CodeQL).
 *     3. Écrire des blocs de requête en utilisant l’opérateur `union` pour combiner
 *        différents ensembles de résultats (fichiers, classes, fonctions, méthodes).
 *     4. Pour chaque type d’élément, sélectionner un libellé (ex. "Fichier", "Classe", etc.),
 *        le nom de l’élément et le chemin relatif dans le projet.
 *
 *   Ces étapes garantissent que toutes les métadonnées essentielles sont extraites pour
 *   permettre une compréhension fine de la structure et du contenu du projet.
 *
 * @kind information
 * @tags metadata
 */

import python

from File f
select "Fichier" as Type, f.getName() as Nom, f.getRelativePath() as Chemin


