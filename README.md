# sql-mutation-generator
An AI tool that creates flawed yet valid SQL queries for data validation

# SQL Mutation Generator

An AI-powered tool that generates incorrect but syntactically valid SQL query variants for testing and educational purposes.

## Features

- Generates 1-5 incorrect variants per input SQL query
- Produces realistic error patterns (join conditions, operators, logic)
- Ensures all variants are executable SQL
- Parallel processing for high throughput
- Fallback mechanism for reliable operation

## Prerequisites

- Python 3.8+
- Together API key 
- Input file with SQL queries (one per line)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sql-mutation-generator.git
cd sql-mutation-generator
