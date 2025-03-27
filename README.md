SQL Mutation Generator
Overview
SQL Mutation Generator is a powerful tool designed to create diverse, executable, but intentionally incorrect SQL queries. This project aims to assist developers and database administrators in testing query validation systems, improving error detection algorithms, and enhancing SQL learning experiences.

Features
Generates multiple erroneous SQL queries from correct ones

Ensures generated queries are executable but yield incorrect results

Provides explanations for each erroneous query

Supports customizable output formats

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/sql-mutation-generator.git
cd sql-mutation-generator
Navigate to the project directory:

bash
cd Wrong_Queries_Generator_Tedros
Open the project in your preferred Python IDE.

Update the file paths in the script to match your system:

Set input_file_path to point to your dev.sql file

Set output_file_path to your desired output location

Usage
Ensure you have the required dependencies installed (sqlite3, together).

Run the script:

bash
python Wrong_Queries_Generator_Tedros.py
The script will generate erroneous queries and save them to the specified output file.

Configuration
Adjust the num_queries parameter in generate_erroneous_queries() to control the number of erroneous queries generated per correct query.

Modify the temperature parameter in the Together AI API call to adjust the creativity of generated queries.


