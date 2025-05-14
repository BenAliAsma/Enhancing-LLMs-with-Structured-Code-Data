# 🧠 Python LSIF Indexer (Deprecated)

> ⚠️ **Notice**: This project is no longer maintained. Please use [`scip-python`](https://github.com/sourcegraph/scip-python) for up-to-date support.

---

## Overview

This repository contains an early implementation of a Language Server Index Format (LSIF) indexer for Python. LSIF provides a standard format for language tools to export information about code, enabling features like "Go to Definition" and "Find References" without needing a live language server.

This implementation adheres to **LSIF spec version 0.4.0** and is intended for experimentation and learning purposes.

---

## Features

* Parses Python workspaces to generate LSIF `.lsif` dump files.
* CLI interface for easy indexing.
* Outputs graph data representing LSIF vertices and edges.
* Includes a **graph visualization dashboard** for inspecting LSIF structure.

![LSIF Graph Visualization](/mnt/data/newplot\(2\).png)

---

## Getting Started

### 🛠 Installation

Requires **Python 3.x**. To install dependencies:

```bash
pip install -r requirements.txt
```

Or using `virtualenv`:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

On macOS (if Python 2.7 is default):

```bash
brew install python@3
```

Use `pip3` if needed.

---

### ▶️ Running the Indexer

Use the provided shell script to run the indexer on a workspace directory:

```bash
./lsif-py lsif_indexer
```

Sample output:

```
Indexing file lsif_indexer/analysis.py
Indexing file lsif_indexer/index.py
...
Processed in 2834.89ms
```

To change output file:

```bash
./lsif-py lsif_indexer -o custom_output.lsif
```

Enable verbose logging with `-v`.

---

## 📊 LSIF Visualization Dashboard

A basic interactive dashboard was created to visualize the LSIF graph data. The graph shows vertices (e.g., `document`, `project`, `metaData`, `$event`) and edges connecting them.

### Features:

* Node coloring based on type/frequency.
* Zoom and pan support.
* Colorbar legend for graph metrics.

---

## ⚰️ Deprecated

This tool is no longer maintained. We recommend migrating to [**scip-python**](https://github.com/sourcegraph/scip-python) for an actively supported LSIF-compatible indexer.

