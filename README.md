# Simple RAG MCP Server

A Retrieval-Augmented Generation (RAG) server built with FastMCP, ChromaDB, LlamaParse, and LlamaIndex.

## Features

- Reads PDF documents from the `data/` directory
- Parses PDFs using LlamaParse
- Stores documents in a persistent ChromaDB collection
- Exposes document ingestion through the Model Context Protocol
- Can be tested with MCP Inspector or VS Code

## Requirements

- Python 3.12+
- `uv`
- Llama Cloud API key
- Node.js 22.19+ for the latest MCP Inspector

## Installation

Clone the repository:

```bash
git clone https://github.com/NishantChaudhary07/simple-rag.git
cd simple-rag
