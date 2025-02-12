## Overview
A Naive attempt of an AI RAG app to try out the new Docling doc processor and NEO4j db. 

Just wanted to try out how Docling works comparing to existing PDF readers as well as Neo4j comparing to chroma db for embeddings storage and chunk relationship if questions are not straightforward or semanitc similarity.

## Features
1. Ollama models for embeddings and query
2. IBM's docling document processor
3. NEO4j db for vector embeddings storage with chunk relationship 
4. Langchain to access llms and docling

## Requirements
Incase if you try it out , there are some pre-requisites
1. Python version below 3.13 , probably create 3.9 venv , langchain docling is not getting downloaded if it is 3.13
2. Neo4j running in local, I've used desktop option. set .env variables for db, user name and password







