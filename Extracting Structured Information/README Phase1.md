# Phase 1 : Extraction d'informations structurées du code à l'aide d'outils d'analyse statique

## Objectif

Cette phase vise à extraire des informations structurées du code source en utilisant des outils d'analyse statique. L'objectif est de capturer la structure, les relations et la sémantique du code pour créer un contexte riche destiné à l'entraînement de modèles de langage (LLMs).

## Outils explorés

### 1. **CodeQL**

CodeQL est un moteur d'analyse statique développé par GitHub, permettant de traiter le code comme des données. Il offre la possibilité d'écrire des requêtes pour identifier des vulnérabilités de sécurité et des motifs indésirables dans le code.

* **Avantages** :

  * Analyse sémantique puissante.
  * Large éventail de requêtes prédéfinies.
  * Supporte plusieurs langages de programmation.

* **Limites** :

  * Principalement axé sur la sécurité.
  * Nécessite la création d'une base de données avant l'analyse.
  * Moins adapté pour une exploration générale du code.

### 2. **Language Server Protocol (LSP)**

LSP est un protocole standardisé permettant aux éditeurs de code d'interagir avec des serveurs de langage, offrant des fonctionnalités telles que l'autocomplétion, la navigation dans le code et la gestion des erreurs.

* **Avantages** :

  * Intégration fluide avec divers éditeurs.
  * Fonctionnalités riches pour l'édition de code.
  * Supporte de nombreux langages via des serveurs dédiés.

* **Limites** :

  * Ne fournit pas directement des métadonnées structurées du code.
  * Dépendant de l'éditeur et du serveur de langage utilisé.

### 3. **Language Server Index Format (LSIF)**

LSIF est un format de données permettant de représenter les informations extraites par un serveur de langage, facilitant ainsi la navigation et l'analyse du code.

* **Avantages** :

  * Permet une exploration efficace du code.
  * Compatible avec divers outils de développement.

* **Limites** :

  * Nécessite la génération préalable d'un index LSIF.
  * Peut être complexe à mettre en place pour de grands projets.

### 4. **multilspy**

multilspy est une bibliothèque Python permettant d'interagir avec des serveurs de langage via LSP, facilitant l'extraction de métadonnées structurées du code.

* **Avantages** :

  * Intégration facile avec des serveurs de langage existants.
  * Permet d'extraire des informations détaillées sur le code.
  * Adapté pour une analyse automatisée.

* **Limites** :

  * Dépendant de la configuration du serveur de langage.
  * Peut nécessiter des ajustements pour des projets spécifiques.

### 5. **SCIP (Semantic Code Index Protocol)**

SCIP est un format de données et un protocole permettant de représenter des informations sémantiques sur le code, facilitant ainsi son exploration et son analyse.

* **Avantages** :

  * Offre une représentation riche et structurée du code.
  * Permet une navigation efficace et une analyse approfondie.
  * Supporte l'intégration avec divers outils d'analyse.

* **Limites** :

  * Nécessite la génération préalable d'un index SCIP.
  * Peut être complexe à mettre en place pour de grands projets.

## Choix final : **SCIP**

Après évaluation des différents outils, le choix s'est porté sur **SCIP** en raison de sa capacité à fournir une représentation sémantique riche et structurée du code, facilitant ainsi son exploration et son analyse. Ce choix permet d'obtenir un contexte détaillé du code, essentiel pour l'entraînement de modèles de langage performants.

## Conclusion

La première phase a permis d'explorer et d'évaluer divers outils d'analyse statique du code. Le choix de SCIP comme outil principal offre une base solide pour l'extraction d'informations structurées, essentielle pour les phases suivantes du projet.
