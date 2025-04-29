import random
import time
import json
import sqlparse
from pathlib import Path
from together import Together
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from sqlparse.sql import Identifier, IdentifierList, Where, Comparison, Parenthesis

# Configuration
input_file_path = r"C:\Users\thago\AppData\Local\Programs\Python\Python38\AI_Assignments\data\dev.sql"
output_file_path = r"C:\Users\thago\AppData\Local\Programs\Python\Python38\AI_Assignments\data\incorrect_queries_output.json"
schema_path = r"C:\Users\thago\AppData\Local\Programs\Python\Python38\AI_Assignments\data\dev_tables.JSON"
api_key = "tgp_v1_aT0A2b-oveRHze-****************"

# Performance tuning
MAX_WORKERS = 3
MIN_VARIATIONS = 1
MAX_VARIATIONS = 5
RETRY_COUNT = 2
TIMEOUT = 30
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Database names to remove from queries
DATABASE_NAMES = {
    'california_schools', 'card_games', 'codebase_community',
    'debit_card_specializing', 'european_football_2', 'financial',
    'formula_1', 'student_club', 'superhero',
    'thrombosis_prediction', 'toxicology'
}

class SchemaAnalyzer:
    def __init__(self, schema_path):
        self.schemas = self._load_all_schemas(schema_path)
    
    def _load_all_schemas(self, path):
        schemas = {}
        for file in Path(path).glob("*.json"):
            try:
                with open(file, 'r') as f:
                    db_id = file.stem
                    schemas[db_id] = json.load(f)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        return schemas
    
    def get_schema(self, db_id):
        return self.schemas.get(db_id)
class QueryParser:
    def __init__(self, schema=None):
        self.schema = schema
        if schema:
            self.table_map = {orig: pretty for orig, pretty in 
                            zip(schema['table_names_original'], schema['table_names'])}
            self.column_map = self._build_column_map()
        else:
            self.table_map = {}
            self.column_map = {}

    def _build_column_map(self):
        column_map = defaultdict(list)
        for col_info in self.schema['column_names_original']:
            if col_info[0] == -1:
                continue
            table_idx, col_name = col_info
            table_name = self.schema['table_names_original'][table_idx]
            column_map[table_name].append(col_name)
        return column_map
    
    def parse(self, query):
        try:
            parsed = sqlparse.parse(query)[0]
            context = {
                'tables': set(),
                'columns': set(),
                'conditions': [],
                'joins': [],
                'parsed': parsed
            }
            self._extract_tables_and_joins(parsed, context)
            self._extract_conditions(parsed, context)
            return context
        except Exception as e:
            print(f"Error parsing query: {str(e)}")
            return None

    def _extract_tables_and_joins(self, parsed, context):
        from_seen = False
        for token in parsed.tokens:
            if token.match(sqlparse.tokens.Keyword, 'FROM'):
                from_seen = True
            elif from_seen and isinstance(token, (Identifier, IdentifierList)):
                identifiers = [token] if isinstance(token, Identifier) else token.get_identifiers()
                for ident in identifiers:
                    table_name = ident.get_real_name()
                    context['tables'].add(table_name)
                    if self.schema and table_name in self.column_map:
                        for col in self.column_map[table_name]:
                            context['columns'].add(f"{table_name}.{col}")
            
            if isinstance(token, sqlparse.sql.Identifier) and token.parent and \
               any(kw in token.parent.value.upper() for kw in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']):
                join_type = token.parent.value.split()[0].upper()
                context['joins'].append({
                    'type': join_type,
                    'table': token.get_real_name(),
                    'raw': token.parent.value
                })

    def _extract_conditions(self, parsed, context):
        for token in parsed.tokens:
            if isinstance(token, Where):
                for sub_token in token.tokens:
                    if isinstance(sub_token, Comparison):
                        left = sub_token.left.value.strip()
                        op = sub_token.token_next(0)[1].value
                        right = sub_token.right.value.strip()
                        context['conditions'].append({
                            'column': left,
                            'operator': op,
                            'value': right,
                            'raw': sub_token.value
                        })
class QueryModifier:
    def __init__(self, schema_analyzer):
        self.schema_analyzer = schema_analyzer
        self.parser = QueryParser()

    def generate_variations(self, query):
        variations = []
        clean_q = clean_query(query)
        query_upper = clean_q.upper()

        # Generate a random number of variations (1 to MAX_VARIATIONS)
        num_variations = random.randint(MIN_VARIATIONS, MAX_VARIATIONS)

        # List of all possible modifications
        modifications = [
            self._swap_and_or,
            self._change_comparison_operators,
            self._modify_join_type,
            self._add_modify_limit,
            self._remove_where_clause,
            self._modify_group_by,
            self._modify_order_by,
            self._add_remove_distinct,
            self._modify_having, 
            self._modify_aggregate,
            self._modify_subquery,
        ]

       

        # Shuffle the modifications to ensure diversity
        random.shuffle(modifications)

        # Apply modifications until we have enough variations
        for mod_func in modifications:
            if len(variations) >= num_variations:
                break
            variation = mod_func(clean_q, query_upper)
            if variation:
                variations.append(variation)

        return variations

    def _swap_and_or(self, clean_q, query_upper):
        if 'WHERE' in query_upper:
            if ' AND ' in query_upper:
                modified = clean_q.replace(' AND ', ' OR ')
                return f"-- Swapped AND/OR operators\n{modified}"
            elif ' OR ' in query_upper:
                modified = clean_q.replace(' OR ', ' AND ')
                return f"-- Swapped AND/OR operators\n{modified}"
        return None

    def _change_comparison_operators(self, clean_q, query_upper):
        ops_map = {'=': '!=', '!=': '=', '>': '<', '<': '>', '>=': '<=', '<=': '>='}
        for old_op, new_op in ops_map.items():
            if old_op in query_upper:
                modified = clean_q.replace(old_op, new_op)
                return f"-- Changed comparison operator ({old_op} to {new_op})\n{modified}"
        return None

    def _modify_join_type(self, clean_q, query_upper):
        if 'INNER JOIN' in query_upper:
            join_type = random.choice(['LEFT'])#, 'RIGHT', 'FULL'])
            modified = clean_q.replace('INNER JOIN', f"{join_type} JOIN")
            return f"-- Changed INNER JOIN to {join_type} JOIN\n{modified}"
        return None

    def _add_modify_limit(self, clean_q, query_upper):
        if 'LIMIT' not in query_upper:
            new_limit = str(random.randint(1, 10))  # Add a reasonable limit
            modified = clean_q + f" LIMIT {new_limit}"
            return f"-- Added LIMIT clause (set to {new_limit})\n{modified}"
        return None

    def _remove_where_clause(self, clean_q, query_upper):
        if 'WHERE' in query_upper:
            where_pos = query_upper.find('WHERE')
            # Preserve the rest of the query (GROUP BY, ORDER BY, LIMIT)
            modified = clean_q[:where_pos].strip()
            return f"-- Removed WHERE clause\n{modified}"
        return None

    def _modify_group_by(self, clean_q, query_upper):
        if 'GROUP BY' in query_upper:
            # Find the GROUP BY clause and replace it with positional reference
            group_by_pos = query_upper.find('GROUP BY')
            # Extract the column list after GROUP BY
            after_group_by = clean_q[group_by_pos + 9:].strip()
            # Replace with positional reference
            modified = clean_q[:group_by_pos] + "GROUP BY 1"
            return f"-- Changed GROUP BY columns to positional reference\n{modified}"
        return None

    def _modify_order_by(self, clean_q, query_upper):
        if 'ORDER BY' in query_upper:
            if 'ASC' in query_upper:
                modified = clean_q.replace('ASC', 'DESC')
                return f"-- Changed ORDER BY ASC to DESC\n{modified}"
            elif 'DESC' in query_upper:
                modified = clean_q.replace('DESC', 'ASC')
                return f"-- Changed ORDER BY DESC to ASC\n{modified}"
        return None

    def _add_remove_distinct(self, clean_q, query_upper):
        # Skip if query contains aggregate functions
        if any(fn in query_upper for fn in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(']):
            return None

        if 'SELECT DISTINCT' in query_upper:
            modified = clean_q.replace('DISTINCT ', '')
            return f"-- Removed DISTINCT\n{modified}"
        else:
            modified = clean_q.replace('SELECT ', 'SELECT DISTINCT ')
            return f"-- Added DISTINCT\n{modified}"
    def _modify_having(self, clean_q, query_upper):
        if 'HAVING' in query_upper:
            having_pos = query_upper.find('HAVING')
            condition = clean_q[having_pos + 6:].strip()
        
            # Reverse comparison operators
            if '>' in condition:
                modified = clean_q.replace('>', '<')
                return f"-- Changed HAVING condition (reversed comparison)\n{modified}"
            elif '<' in condition:
                modified = clean_q.replace('<', '>')
                return f"-- Changed HAVING condition (reversed comparison)\n{modified}"
        
            # Swap AND/OR operators
            if ' AND ' in condition:
                modified = clean_q.replace(' AND ', ' OR ')
                return f"-- Swapped AND/OR operators in HAVING clause\n{modified}"
            elif ' OR ' in condition:
                modified = clean_q.replace(' OR ', ' AND ')
                return f"-- Swapped AND/OR operators in HAVING clause\n{modified}"
        
            # Remove HAVING clause entirely
            modified = clean_q[:having_pos].strip()
            return f"-- Removed HAVING clause\n{modified}"
        return None


    def _modify_aggregate(self, clean_q, query_upper):
        if 'SUM(' in query_upper:
            modified = clean_q.replace('SUM(', 'AVG(')
            return f"-- Changed SUM to AVG\n{modified}"
        elif 'COUNT(' in query_upper:
            modified = clean_q.replace('COUNT(', 'MAX(')
            return f"-- Changed COUNT to MAX\n{modified}"
        return None
    def _modify_subquery(self, clean_q, query_upper):
        # Find the position of the first subquery
        subquery_start = query_upper.find('(SELECT')
        if subquery_start == -1:
            return None  # No subquery found
    
        # Find the end of the subquery
        subquery_end = query_upper.find(')', subquery_start)
        if subquery_end == -1:
            return None  # Invalid subquery syntax
    
        # Extract the subquery
        subquery = clean_q[subquery_start + 1:subquery_end]
    
        # Change comparison operators
        if '=' in subquery:
            modified_subquery = subquery.replace('=', '!=')
            modified = clean_q.replace(subquery, modified_subquery)
            return f"-- Changed subquery comparison operator (= to !=)\n{modified}"
        elif '!=' in subquery:
            modified_subquery = subquery.replace('!=', '=')
            modified = clean_q.replace(subquery, modified_subquery)
            return f"-- Changed subquery comparison operator (!= to =)\n{modified}"
    
        # Swap AND/OR operators
        if ' AND ' in subquery:
            modified_subquery = subquery.replace(' AND ', ' OR ')
            modified = clean_q.replace(subquery, modified_subquery)
            return f"-- Swapped AND/OR operators in subquery\n{modified}"
        elif ' OR ' in subquery:
            modified_subquery = subquery.replace(' OR ', ' AND ')
            modified = clean_q.replace(subquery, modified_subquery)
            return f"-- Swapped AND/OR operators in subquery\n{modified}"
    
        # Remove subquery entirely
        #modified = clean_q.replace(f"({subquery})", '1')
        #return f"-- Removed subquery\n{modified}"




def clean_query(query):
    """Remove database names and fix formatting issues"""
    query = query.strip().rstrip(';')
    for db_name in DATABASE_NAMES:
        if query.lower().rstrip().endswith(db_name.lower()):
            db_pos = len(query) - len(db_name)
            if db_pos >= 0 and query[db_pos:].lower() == db_name.lower():
                query = query[:db_pos].strip()
                break
    query = query.replace('\\_', '_')
    return ' '.join(query.split())

def read_sample_queries(file_path):
    """Read SQL queries from file and return a balanced sample."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        # Group queries by database
        db_queries = defaultdict(list)
        for query in queries:
            for db_name in DATABASE_NAMES:
                if db_name.lower() in query.lower():
                    db_queries[db_name].append(query)
                    break

        # Sample queries: 9 from each database, 10 from one database
        sampled_queries = []
        for db_name, db_q in db_queries.items():
            if db_name == "debit_card_specializing":
                sampled_queries.extend(random.sample(db_q, min(10, len(db_q))))
            else:
                sampled_queries.extend(random.sample(db_q, min(9, len(db_q))))
        
        # Ensure we have exactly 100 queries
        if len(sampled_queries) > 100:
            sampled_queries = sampled_queries[:100]
        elif len(sampled_queries) < 100:
            # If there aren't enough queries, add more from the largest database
            largest_db = max(db_queries, key=lambda k: len(db_queries[k]))
            additional = random.sample(db_queries[largest_db], 100 - len(sampled_queries))
            sampled_queries.extend(additional)
        
        return sampled_queries
    except Exception as e:
        print(f"Error reading input file: {e}")
        return []

def validate_variations(text, expected_count, original_query):
    """Validate generated variations meet requirements"""
    variations = []
    current = []
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('-- Explanation'):
            if current:
                variations.append('\n'.join(current))
                current = []
            current.append(line)
        elif line and current:
            current.append(line)
    if current:
        variations.append('\n'.join(current))
    
    # Skip validation if the query is unchanged
    clean_original = clean_query(original_query)
    valid_variations = []
    for var in variations[:expected_count]:
        try:
            sql_lines = [l for l in var.split('\n') if l and not l.startswith('--')]
            if not sql_lines:
                continue
            sql_part = clean_query(sql_lines[-1].strip())
            if sql_part == clean_original:
                continue
            valid_variations.append(var)
        except:
            continue
    return '\n\n'.join(valid_variations) if valid_variations else None

def process_query(query, client, schema_analyzer):
    """Process a single query to generate variations"""
    try:
        modifier = QueryModifier(schema_analyzer)
        variations = modifier.generate_variations(query)
        
        if variations:
            return query, variations
        
        # Fallback to API if local modifications fail
        num_variations = random.randint(MIN_VARIATIONS, MAX_VARIATIONS)
        prompt = f"""Generate {num_variations} incorrect but executable SQL variations for:
{query}
Each must:
1. Use only existing tables/columns
2. Be syntactically valid
3. Produce wrong results
4. Include a brief explanation"""

        for attempt in range(RETRY_COUNT):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
                if response.choices:
                    generated = response.choices[0].message.content
                    validated = validate_variations(generated, num_variations, query)
                    if validated:
                        return query, validated
            except Exception as e:
                print(f"API error (attempt {attempt+1}): {str(e)}")
                time.sleep(1)
        
        return query, "-- No valid variations generated"
    except Exception as e:
        return query, f"-- Processing error: {str(e)}"

def main():
    print("Starting SQL variation generation...")
    start_time = time.time()
    
    # Initialize services
    client = Together(api_key=api_key)
    schema_analyzer = SchemaAnalyzer(schema_path)
    
    # Read and validate input queries
    queries = read_sample_queries(input_file_path)
    if not queries:
        print("Error: No valid SQL queries found in input file")
        return
    
    # Group queries by database
    db_queries = defaultdict(list)
    for query in queries:
        for db_name in DATABASE_NAMES:
            if db_name.lower() in query.lower():
                db_queries[db_name].append(query)
                break
    
    # Ensure all databases are represented
    results = []
    for db_name, db_q in db_queries.items():
        sample_size = min(10, len(db_q))  # Sample up to 10 queries per database
        sampled_queries = random.sample(db_q, sample_size)
        
        # Process sampled queries
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_query, q, client, schema_analyzer): q 
                for q in sampled_queries
            }
            
            for future in futures:
                try:
                    results.append(future.result(timeout=TIMEOUT))
                except Exception as e:
                    results.append((futures[future], f"-- Error: {str(e)}"))
    
    # Write results with the desired format
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for i, (query, variants) in enumerate(results, 1):
            f.write(f"-- Original query {i}:\n-- {query}\n\n")
            if isinstance(variants, list):
                for j, var in enumerate(variants, 1):
                    # Fix: Escape backslashes in the explanation
                    explanation = var.split('-- ')[1].split('\n')[0].replace('\\', '\\\\')
                    f.write(f"-- Explanation {j}: {explanation}\n")
                    # Fix: Escape backslashes in the query
                    query_line = var.split('\n')[1].replace('\\', '\\\\')
                    f.write(f"{query_line}\n\n")
            else:
                f.write(f"{variants}\n\n")
            f.write("=" * 80 + "\n\n")
    
    print(f"\nCompleted in {time.time() - start_time:.2f} seconds")
    print(f"Results written to {output_file_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        print("Script execution completed")

