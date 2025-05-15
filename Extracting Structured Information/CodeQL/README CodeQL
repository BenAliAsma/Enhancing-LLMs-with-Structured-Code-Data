# Phase 1: Extracting Structured Information from Code Using CodeQL

## 📌 Overview

This project constitutes **Phase 1** of a broader internship initiative focused on using **static analysis tools** to extract structured metadata from source code. The objective is to capture details like class hierarchies, method signatures, variable types, and dependencies to create a rich context suitable for downstream applications, such as large language models (LLMs).

In this phase, **CodeQL** was selected as the primary tool due to its powerful querying capabilities and support for detailed static code analysis. Other tools considered during the preliminary survey included **SonarQube** and **Language Server Protocol (LSP)**.

---

## ⚙️ What This Phase Includes

- Cloning and preparing a target GitHub repository
- Creating a CodeQL database from the repository source code
- Running CodeQL analysis on the database
- Exporting results in SARIF format and converting them to JSON for easier processing

---

## 🧰 Prerequisites

Make sure the following are installed:

- [CodeQL CLI](https://github.com/github/codeql-cli-binaries)
- [Git](https://git-scm.com/)
- [jq](https://stedolan.github.io/jq/) (for converting SARIF to JSON)
- Python (if analyzing Python code)

---

## 🚀 Usage

To analyze a GitHub repository using the automated batch script:

```bash
analyze.bat [GitHub Repo URL]
````

### Example:

```bash
analyze.bat https://github.com/astropy/astropy
```

### Output Files:

* `resultats_astropy.sarif`: Raw analysis output in SARIF format.
* `resultats_astropy.json`: Processed JSON version of the SARIF data.

---

## 📂 Script Breakdown (`analyze.bat`)

```batch
1. Checks if a GitHub URL is provided.
2. Clones the given repository.
3. Creates a CodeQL database using the source code.
4. Analyzes the codebase for security and quality issues.
5. Converts SARIF output into JSON format.
```

---

## 📝 Notes

* The script is currently tailored for **Python** repositories. Update the `--language` parameter in the script for other languages.
* The default repo name in the script is `astropy-main`; modify this if analyzing a different project.
* This phase sets the foundation for generating contextual knowledge usable by LLMs in subsequent stages of the project.

