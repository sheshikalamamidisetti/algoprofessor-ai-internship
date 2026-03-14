#!/bin/bash
# ============================================================
# SETUP GUIDE
# Day 15: Graph RAG + Advanced RAG Techniques
# Author: Sheshikala
# ============================================================

echo "=============================================="
echo "DAY 15 SETUP GUIDE"
echo "Graph RAG + Advanced RAG Techniques"
echo "=============================================="

echo ""
echo "Step 1: Install dependencies"
pip install -r requirements.txt

echo ""
echo "Step 2: Run Graph RAG (works without Neo4j)"
echo "  python graph_rag.py"

echo ""
echo "Step 3: Run Advanced RAG techniques"
echo "  python advanced_rag.py"

echo ""
echo "Step 4: Run Streaming QA App"
echo "  python streaming_qa_app.py"

echo ""
echo "Step 5: Open evaluation notebook"
echo "  jupyter notebook day15_eval.ipynb"

echo ""
echo "Optional: Use real Neo4j"
echo "  Install Neo4j from https://neo4j.com/download/"
echo "  Start Neo4j service and update URI in graph_rag.py"
echo "  Default URI: bolt://localhost:7687"

echo ""
echo "=============================================="
echo "All files run without Neo4j using mock graph."
echo "No API keys required unless using Groq."
echo "=============================================="
