import random
import time
from together import Together
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configuration
input_file_path = r"C:\Users\thago\AppData\Local\Programs\Python\Python38\AI_Assignments\data\dev.sql"
output_file_path = r"C:\Users\thago\AppData\Local\Programs\Python\Python38\AI_Assignments\data\incorrect_queries.JSON"
api_key = "tgp_v1_GZpxfMoywN9UP3btHjZ8q6bsyMyGd6bY-lIr6i7AQpw"

# Performance tuning
MAX_WORKERS = 3
MIN_VARIATIONS = 1
MAX_VARIATIONS = 5
RETRY_COUNT = 2  # Reduced retries for faster failure
TIMEOUT = 30  # Increased timeout for complex queries
MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def read_sample_queries(file_path, sample_size=100):
    """Read SQL queries from file and return a sample with validation."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [q.strip() for q in f.readlines() if q.strip() and 
                      any(keyword in q.upper() for keyword in ['SELECT', 'FROM'])]
        print(f"Loaded {len(queries)} valid queries, selecting {min(sample_size, len(queries))} samples")
        return random.sample(queries, min(sample_size, len(queries)))
    except Exception as e:
        print(f"Error reading input file: {e}")
        return []

def generate_variations(query, client):
    """Generate incorrect SQL variations with enhanced prompt engineering."""
    num_variations = random.randint(MIN_VARIATIONS, MAX_VARIATIONS)
    prompt = f"""
    Given this correct SQL query:
    {query}

    Generate exactly {num_variations} incorrect but executable variations that:
    1. Use ONLY the existing tables and columns from the original query
    2. Are syntactically valid SQL that will execute without errors
    3. Produce incorrect results through logical flaws
    4. Each has a different type of error from this list:
       - Wrong comparison operators (=, !=, >, <, LIKE)
       - Incorrect values in conditions
       - Modified JOIN conditions
       - Changed ORDER BY direction
       - Wrong LIMIT/OFFSET values
       - Added/removed WHERE conditions
       - Swapped AND/OR operators

    Format each variation EXACTLY like this:
    -- Explanation [N]: [specific reason why wrong]
    [executable SQL query]

    Example for a different query:
    -- Explanation 1: Changed condition value from 'Fast' to 'Slow'
    SELECT DISTINCT t1.team_long_name FROM Team AS t1 
    INNER JOIN Team_Attributes AS t2 ON t1.team_api_id = t2.team_api_id 
    WHERE t2.buildUpPlaySpeedClass = 'Slow'
    """

    for attempt in range(RETRY_COUNT):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,  # Increased for better quality
                temperature=0.7,
                top_p=0.8,
            )
            
            if response.choices:
                generated = response.choices[0].message.content
                validated = validate_variations(generated, num_variations, query)
                if validated:
                    return validated
                elif attempt == RETRY_COUNT - 1:
                    return generate_fallback_variations(query, num_variations)
        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {str(e)}")
            time.sleep(1)  # Brief delay before retry
    
    return generate_fallback_variations(query, num_variations)

def generate_fallback_variations(query, num_variations):
    """Generate simple variations when API fails."""
    variations = []
    base_errors = [
        ("Changed = to !=", query.replace("=", "!=")),
        ("Added false condition", query + " AND 1=0" if "WHERE" in query 
         else query + " WHERE 1=0" if "FROM" in query else query),
        ("Reversed ORDER BY", 
         query.replace("ASC", "DESC") if "ASC" in query 
         else query.replace("DESC", "ASC") if "DESC" in query 
         else query + " ORDER BY 1 DESC" if "FROM" in query else query),
        ("Limited to 1 row", 
         query.replace("LIMIT", "LIMIT 1") if "LIMIT" in query 
         else query + " LIMIT 1" if "FROM" in query else query)
    ]
    
    for i, (desc, variation) in enumerate(base_errors[:num_variations], 1):
        variations.append(f"-- Explanation {i}: {desc}\n{variation}")
    
    return '\n\n'.join(variations) if variations else "-- Failed to generate variations"

def validate_variations(text, expected_count, original_query):
    """Strict validation of generated variations."""
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
    
    # Validate each variation
    valid_variations = []
    for i, var in enumerate(variations[:expected_count], 1):
        try:
            if not var.startswith('-- Explanation'):
                continue
                
            # Ensure proper numbering
            if f"-- Explanation {i}:" not in var:
                var = var.replace("-- Explanation:", f"-- Explanation {i}:", 1)
            
            # Check SQL syntax
            sql_part = var.split('\n')[-1]
            if not sql_part.strip().upper().startswith(('SELECT', 'WITH')):
                continue
                
            # Verify it's different from original
            if sql_part.strip() == original_query.strip():
                continue
                
            valid_variations.append(var)
        except:
            continue
    
    return '\n\n'.join(valid_variations) if valid_variations else None

def process_query(query, client):
    """Wrapper function with enhanced error handling."""
    try:
        result = generate_variations(query, client)
        return query, result or "-- No valid variations generated"
    except Exception as e:
        return query, f"-- Processing error: {str(e)}"

def main():
    print("Starting SQL mutation process with robust error handling...")
    start_time = time.time()

    # Initialize client
    client = Together(api_key=api_key)

    # Read and validate queries
    sample_queries = read_sample_queries(input_file_path, 100)
    if not sample_queries:
        print("No valid queries found to process")
        return

    print(f"Processing {len(sample_queries)} queries with {MAX_WORKERS} workers...")

    # Process queries with ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_query, q, client): q for q in sample_queries}
        for future in tqdm(futures, total=len(sample_queries), desc="Generating variants"):
            try:
                results.append(future.result(timeout=TIMEOUT))
            except Exception as e:
                results.append((futures[future], f"-- Timeout/error: {str(e)}"))

    # Write results with improved formatting
    with open(output_file_path, 'w', encoding='utf-8') as f:
        success_count = 0
        for i, (query, variants) in enumerate(results, 1):
            f.write(f"-- Original query {i}:\n-- {query}\n\n")
            if variants.startswith("--"):
                f.write(variants + "\n")
            else:
                success_count += 1
                f.write(variants + "\n")
            f.write("\n" + ("=" * 80) + "\n\n")

    total_time = time.time() - start_time
    success_rate = (success_count / len(sample_queries)) * 100
    print(f"\nCompleted in {total_time:.2f} seconds")
    print(f"Success rate: {success_rate:.1f}% of queries got valid variations")
    print(f"Results written to {output_file_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Script execution completed")
