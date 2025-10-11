#!/usr/bin/env python3
"""
sql_to_nl.py

Convert SQL queries to natural language descriptions.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Natural language templates for SQL operations
NL_TEMPLATES = {
    'select_all': "Show all records from {table}",
    'select_columns': "Show {columns} from {table}",
    'select_where': "Show {columns} from {table} where {condition}",
    'count': "Count the number of records in {table}",
    'count_where': "Count the number of records in {table} where {condition}",
    'sum': "Calculate the sum of {column} in {table}",
    'avg': "Calculate the average {column} in {table}",
    'max': "Find the maximum {column} in {table}",
    'min': "Find the minimum {column} in {table}",
    'group_by': "Show {columns} grouped by {group_columns} from {table}",
    'order_by': "Show {columns} from {table} sorted by {order_column} {direction}",
    'join': "Combine data from {tables} where {join_condition}",
    'insert': "Add a new record to {table} with {assignments}",
    'update': "Update records in {table} setting {assignments} where {condition}",
    'delete': "Delete records from {table} where {condition}"
}

# SQL operator mappings to natural language
OPERATOR_MAPPINGS = {
    '=': 'equals',
    '!=': 'does not equal',
    '<>': 'does not equal',
    '>': 'is greater than',
    '<': 'is less than',
    '>=': 'is greater than or equal to',
    '<=': 'is less than or equal to',
    'LIKE': 'contains',
    'NOT LIKE': 'does not contain',
    'IN': 'is one of',
    'NOT IN': 'is not one of',
    'BETWEEN': 'is between',
    'IS NULL': 'is empty',
    'IS NOT NULL': 'is not empty'
}

# SQL functions to natural language
FUNCTION_MAPPINGS = {
    'COUNT': 'count',
    'SUM': 'sum',
    'AVG': 'average',
    'MAX': 'maximum',
    'MIN': 'minimum',
    'UPPER': 'uppercase',
    'LOWER': 'lowercase',
    'LENGTH': 'length of',
    'DISTINCT': 'unique'
}

class SQLToNLConverter:
    def __init__(self):
        self.parsed_query = {}
    
    def parse_sql(self, sql: str) -> Dict[str, Any]:
        """Parse SQL query into components"""
        sql = sql.strip().rstrip(';')
        sql_upper = sql.upper()
        
        parsed = {
            'type': 'unknown',
            'tables': [],
            'columns': [],
            'where': '',
            'joins': [],
            'group_by': [],
            'order_by': [],
            'having': '',
            'functions': [],
            'original': sql
        }
        
        # Determine query type
        if sql_upper.startswith('SELECT'):
            parsed['type'] = 'select'
            self.parse_select(sql, parsed)
        elif sql_upper.startswith('INSERT'):
            parsed['type'] = 'insert'
            self.parse_insert(sql, parsed)
        elif sql_upper.startswith('UPDATE'):
            parsed['type'] = 'update'
            self.parse_update(sql, parsed)
        elif sql_upper.startswith('DELETE'):
            parsed['type'] = 'delete'
            self.parse_delete(sql, parsed)
        
        return parsed
    
    def parse_select(self, sql: str, parsed: Dict[str, Any]):
        """Parse SELECT statement"""
        sql_upper = sql.upper()
        
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE)
        if select_match:
            select_clause = select_match.group(1)
            parsed['columns'] = self.parse_columns(select_clause)
        
        # Extract FROM clause
        from_match = re.search(r'FROM\s+([\w\s,]+?)(?:\s+WHERE|\s+JOIN|\s+GROUP|\s+ORDER|\s+HAVING|$)', sql, re.IGNORECASE)
        if from_match:
            from_clause = from_match.group(1)
            parsed['tables'] = [t.strip() for t in from_clause.split(',')]
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP|\s+ORDER|\s+HAVING|$)', sql, re.IGNORECASE)
        if where_match:
            parsed['where'] = where_match.group(1).strip()
        
        # Extract JOIN clauses
        join_matches = re.finditer(r'((?:INNER|LEFT|RIGHT|FULL)?\s*JOIN)\s+([\w]+)\s+ON\s+(.*?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+HAVING|$)', sql, re.IGNORECASE)
        for match in join_matches:
            join_type = match.group(1).strip()
            table = match.group(2).strip()
            condition = match.group(3).strip()
            parsed['joins'].append({
                'type': join_type,
                'table': table,
                'condition': condition
            })
        
        # Extract GROUP BY
        group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+ORDER|\s+HAVING|$)', sql, re.IGNORECASE)
        if group_match:
            parsed['group_by'] = [col.strip() for col in group_match.group(1).split(',')]
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)', sql, re.IGNORECASE)
        if order_match:
            order_clause = order_match.group(1)
            parsed['order_by'] = self.parse_order_by(order_clause)
        
        # Extract HAVING
        having_match = re.search(r'HAVING\s+(.*?)(?:\s+ORDER|$)', sql, re.IGNORECASE)
        if having_match:
            parsed['having'] = having_match.group(1).strip()
        
        # Detect aggregate functions
        for func in FUNCTION_MAPPINGS.keys():
            if func in sql_upper:
                parsed['functions'].append(func)
    
    def parse_columns(self, select_clause: str) -> List[str]:
        """Parse column list from SELECT clause"""
        if select_clause.strip() == '*':
            return ['*']
        
        # Split by comma, but be careful of function calls
        columns = []
        current_col = ""
        paren_count = 0
        
        for char in select_clause:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                columns.append(current_col.strip())
                current_col = ""
                continue
            
            current_col += char
        
        if current_col.strip():
            columns.append(current_col.strip())
        
        return columns
    
    def parse_order_by(self, order_clause: str) -> List[Dict[str, str]]:
        """Parse ORDER BY clause"""
        order_items = []
        for item in order_clause.split(','):
            item = item.strip()
            if item.upper().endswith(' DESC'):
                column = item[:-5].strip()
                direction = 'DESC'
            elif item.upper().endswith(' ASC'):
                column = item[:-4].strip()
                direction = 'ASC'
            else:
                column = item
                direction = 'ASC'
            
            order_items.append({
                'column': column,
                'direction': direction
            })
        
        return order_items
    
    def parse_insert(self, sql: str, parsed: Dict[str, Any]):
        """Parse INSERT statement"""
        # Extract table name
        table_match = re.search(r'INSERT\s+INTO\s+([\w]+)', sql, re.IGNORECASE)
        if table_match:
            parsed['tables'] = [table_match.group(1)]
        
        # Extract columns and values
        columns_match = re.search(r'\((.*?)\)\s+VALUES', sql, re.IGNORECASE)
        if columns_match:
            parsed['columns'] = [col.strip() for col in columns_match.group(1).split(',')]
    
    def parse_update(self, sql: str, parsed: Dict[str, Any]):
        """Parse UPDATE statement"""
        # Extract table name
        table_match = re.search(r'UPDATE\s+([\w]+)', sql, re.IGNORECASE)
        if table_match:
            parsed['tables'] = [table_match.group(1)]
        
        # Extract SET clause
        set_match = re.search(r'SET\s+(.*?)(?:\s+WHERE|$)', sql, re.IGNORECASE)
        if set_match:
            parsed['set_clause'] = set_match.group(1).strip()
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)$', sql, re.IGNORECASE)
        if where_match:
            parsed['where'] = where_match.group(1).strip()
    
    def parse_delete(self, sql: str, parsed: Dict[str, Any]):
        """Parse DELETE statement"""
        # Extract table name
        table_match = re.search(r'DELETE\s+FROM\s+([\w]+)', sql, re.IGNORECASE)
        if table_match:
            parsed['tables'] = [table_match.group(1)]
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)$', sql, re.IGNORECASE)
        if where_match:
            parsed['where'] = where_match.group(1).strip()
    
    def convert_condition_to_nl(self, condition: str) -> str:
        """Convert SQL condition to natural language"""
        if not condition:
            return ""
        
        # Handle common operators
        for sql_op, nl_op in OPERATOR_MAPPINGS.items():
            if sql_op in condition.upper():
                condition = re.sub(
                    rf'\b{re.escape(sql_op)}\b',
                    nl_op,
                    condition,
                    flags=re.IGNORECASE
                )
        
        # Handle LIKE patterns
        condition = re.sub(r"LIKE\s+'%([^%]+)%'", r"contains '\1'", condition, flags=re.IGNORECASE)
        condition = re.sub(r"LIKE\s+'([^%]+)%'", r"starts with '\1'", condition, flags=re.IGNORECASE)
        condition = re.sub(r"LIKE\s+'%([^%]+)'", r"ends with '\1'", condition, flags=re.IGNORECASE)
        
        # Handle BETWEEN
        condition = re.sub(r'(\w+)\s+BETWEEN\s+(\w+)\s+AND\s+(\w+)', 
                          r'\1 is between \2 and \3', condition, flags=re.IGNORECASE)
        
        # Handle IN clauses
        condition = re.sub(r'(\w+)\s+IN\s+\((.*?)\)', 
                          r'\1 is one of (\2)', condition, flags=re.IGNORECASE)
        
        return condition
    
    def format_columns_for_nl(self, columns: List[str]) -> str:
        """Format column list for natural language"""
        if not columns or columns == ['*']:
            return "all columns"
        
        # Clean up column names (remove aliases, functions)
        clean_columns = []
        for col in columns:
            # Remove AS aliases
            col = re.sub(r'\s+AS\s+\w+', '', col, flags=re.IGNORECASE)
            
            # Extract column name from functions
            func_match = re.search(r'(\w+)\((.*?)\)', col)
            if func_match:
                func_name = func_match.group(1).upper()
                inner_col = func_match.group(2)
                if func_name in FUNCTION_MAPPINGS:
                    nl_func = FUNCTION_MAPPINGS[func_name]
                    clean_columns.append(f"{nl_func} of {inner_col}")
                else:
                    clean_columns.append(col)
            else:
                clean_columns.append(col)
        
        if len(clean_columns) == 1:
            return clean_columns[0]
        elif len(clean_columns) == 2:
            return f"{clean_columns[0]} and {clean_columns[1]}"
        else:
            return ", ".join(clean_columns[:-1]) + f", and {clean_columns[-1]}"
    
    def convert_to_nl(self, sql: str) -> Dict[str, Any]:
        """Convert SQL query to natural language"""
        parsed = self.parse_sql(sql)
        
        if parsed['type'] == 'select':
            return self.convert_select_to_nl(parsed)
        elif parsed['type'] == 'insert':
            return self.convert_insert_to_nl(parsed)
        elif parsed['type'] == 'update':
            return self.convert_update_to_nl(parsed)
        elif parsed['type'] == 'delete':
            return self.convert_delete_to_nl(parsed)
        else:
            return {
                'natural_language': f"Execute the SQL query: {sql}",
                'confidence': 0.3,
                'components': parsed
            }
    
    def convert_select_to_nl(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert SELECT query to natural language"""
        table = parsed['tables'][0] if parsed['tables'] else 'table'
        columns_nl = self.format_columns_for_nl(parsed['columns'])
        
        # Base description
        if parsed['functions'] and 'COUNT' in parsed['functions']:
            if parsed['where']:
                condition_nl = self.convert_condition_to_nl(parsed['where'])
                nl = f"Count the number of records in {table} where {condition_nl}"
            else:
                nl = f"Count the number of records in {table}"
        
        elif any(func in parsed['functions'] for func in ['SUM', 'AVG', 'MAX', 'MIN']):
            func = next(func for func in parsed['functions'] if func in ['SUM', 'AVG', 'MAX', 'MIN'])
            func_nl = FUNCTION_MAPPINGS[func]
            column = parsed['columns'][0] if parsed['columns'] else 'column'
            # Extract column from function call
            func_match = re.search(r'\((.*?)\)', column)
            if func_match:
                column = func_match.group(1)
            
            nl = f"Calculate the {func_nl} of {column} in {table}"
            
            if parsed['where']:
                condition_nl = self.convert_condition_to_nl(parsed['where'])
                nl += f" where {condition_nl}"
        
        else:
            if parsed['where']:
                condition_nl = self.convert_condition_to_nl(parsed['where'])
                nl = f"Show {columns_nl} from {table} where {condition_nl}"
            else:
                if columns_nl == "all columns":
                    nl = f"Show all records from {table}"
                else:
                    nl = f"Show {columns_nl} from {table}"
        
        # Add JOIN information
        if parsed['joins']:
            join_descriptions = []
            for join in parsed['joins']:
                join_table = join['table']
                join_condition = self.convert_condition_to_nl(join['condition'])
                join_descriptions.append(f"combined with {join_table} where {join_condition}")
            nl += " " + ", ".join(join_descriptions)
        
        # Add GROUP BY information
        if parsed['group_by']:
            group_cols = ", ".join(parsed['group_by'])
            nl += f" grouped by {group_cols}"
        
        # Add ORDER BY information
        if parsed['order_by']:
            order_descriptions = []
            for order in parsed['order_by']:
                direction = "descending" if order['direction'] == 'DESC' else "ascending"
                order_descriptions.append(f"{order['column']} {direction}")
            order_desc = ", ".join(order_descriptions)
            nl += f" sorted by {order_desc}"
        
        # Add HAVING information
        if parsed['having']:
            having_nl = self.convert_condition_to_nl(parsed['having'])
            nl += f" having {having_nl}"
        
        # Calculate confidence
        confidence = self.calculate_confidence(parsed)
        
        return {
            'natural_language': nl,
            'confidence': confidence,
            'components': parsed
        }
    
    def convert_insert_to_nl(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert INSERT query to natural language"""
        table = parsed['tables'][0] if parsed['tables'] else 'table'
        
        if parsed['columns']:
            columns = ", ".join(parsed['columns'])
            nl = f"Add a new record to {table} with values for {columns}"
        else:
            nl = f"Add a new record to {table}"
        
        return {
            'natural_language': nl,
            'confidence': 0.8,
            'components': parsed
        }
    
    def convert_update_to_nl(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert UPDATE query to natural language"""
        table = parsed['tables'][0] if parsed['tables'] else 'table'
        
        nl = f"Update records in {table}"
        
        if parsed.get('set_clause'):
            nl += f" setting {parsed['set_clause']}"
        
        if parsed['where']:
            condition_nl = self.convert_condition_to_nl(parsed['where'])
            nl += f" where {condition_nl}"
        
        return {
            'natural_language': nl,
            'confidence': 0.8,
            'components': parsed
        }
    
    def convert_delete_to_nl(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DELETE query to natural language"""
        table = parsed['tables'][0] if parsed['tables'] else 'table'
        
        if parsed['where']:
            condition_nl = self.convert_condition_to_nl(parsed['where'])
            nl = f"Delete records from {table} where {condition_nl}"
        else:
            nl = f"Delete all records from {table}"
        
        return {
            'natural_language': nl,
            'confidence': 0.8,
            'components': parsed
        }
    
    def calculate_confidence(self, parsed: Dict[str, Any]) -> float:
        """Calculate confidence score for the conversion"""
        confidence = 0.7  # Base confidence
        
        # Increase confidence for recognized patterns
        if parsed['type'] in ['select', 'insert', 'update', 'delete']:
            confidence += 0.1
        
        # Decrease confidence for complex queries
        if len(parsed['joins']) > 1:
            confidence -= 0.1
        
        if parsed['having']:
            confidence -= 0.1
        
        # Increase confidence for simple queries
        if not parsed['joins'] and not parsed['group_by'] and not parsed['having']:
            confidence += 0.1
        
        return min(1.0, confidence)

def main():
    parser = argparse.ArgumentParser(description="Convert SQL queries to natural language")
    parser.add_argument('query', nargs='?', help='SQL query to convert')
    parser.add_argument('--input', help='File with SQL queries to convert')
    parser.add_argument('--output', help='Output file for natural language descriptions')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    converter = SQLToNLConverter()
    
    if args.interactive:
        print("\nSQL to Natural Language Converter")
        print("Enter 'quit' to exit")
        
        while True:
            try:
                query = input("\nEnter SQL query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                result = converter.convert_to_nl(query)
                
                print(f"\nNatural Language: {result['natural_language']}")
                print(f"Confidence: {result['confidence']:.2f}")
                
                if result['confidence'] < 0.7:
                    print("⚠️  Low confidence - please verify the description")
                
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
    
    elif args.input:
        print(f"Converting SQL queries from {args.input}...")
        
        with open(args.input, 'r', encoding='utf-8') as f:
            if args.input.endswith('.jsonl'):
                queries = [json.loads(line) for line in f]
            else:
                data = json.load(f)
                queries = data if isinstance(data, list) else [data]
        
        results = []
        for item in queries:
            sql_query = item.get('sql', str(item))
            result = converter.convert_to_nl(sql_query)
            result['original_sql'] = sql_query
            results.append(result)
        
        print(f"Converted {len(results)} queries")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        else:
            for i, result in enumerate(results[:5]):
                print(f"\n{i+1}. SQL: {result['original_sql']}")
                print(f"   NL: {result['natural_language']}")
                print(f"   Confidence: {result['confidence']:.2f}")
    
    elif args.query:
        result = converter.convert_to_nl(args.query)
        
        print(f"SQL: {args.query}")
        print(f"Natural Language: {result['natural_language']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        if result['confidence'] < 0.7:
            print("⚠️  Low confidence - please verify the description")
    
    else:
        # Show examples
        print("SQL to Natural Language Converter")
        print("\nExample conversions:")
        examples = [
            "SELECT * FROM users",
            "SELECT name, email FROM users WHERE age > 25",
            "SELECT COUNT(*) FROM orders",
            "SELECT AVG(amount) FROM orders WHERE date > '2023-01-01'",
            "UPDATE users SET city = 'New York' WHERE id = 1"
        ]
        
        for sql in examples:
            result = converter.convert_to_nl(sql)
            print(f"\nSQL: {sql}")
            print(f"NL:  {result['natural_language']}")

if __name__ == "__main__":
    main()