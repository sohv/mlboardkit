#!/usr/bin/env python3
"""
nl_to_sql.py

Convert natural language queries to SQL.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# SQL templates for common operations
SQL_TEMPLATES = {
    'select_all': "SELECT * FROM {table}",
    'select_columns': "SELECT {columns} FROM {table}",
    'select_where': "SELECT {columns} FROM {table} WHERE {condition}",
    'count': "SELECT COUNT(*) FROM {table}",
    'count_where': "SELECT COUNT(*) FROM {table} WHERE {condition}",
    'group_by': "SELECT {columns}, COUNT(*) FROM {table} GROUP BY {group_columns}",
    'order_by': "SELECT {columns} FROM {table} ORDER BY {order_column} {direction}",
    'join': "SELECT {columns} FROM {table1} JOIN {table2} ON {join_condition}",
    'aggregate': "SELECT {agg_function}({column}) FROM {table}",
    'insert': "INSERT INTO {table} ({columns}) VALUES ({values})",
    'update': "UPDATE {table} SET {assignments} WHERE {condition}",
    'delete': "DELETE FROM {table} WHERE {condition}"
}

# Common SQL keywords and patterns
SQL_KEYWORDS = {
    'select': ['show', 'display', 'get', 'find', 'list', 'retrieve'],
    'where': ['where', 'with', 'having', 'that have', 'filter by'],
    'count': ['count', 'number of', 'how many', 'total'],
    'sum': ['sum', 'total', 'add up'],
    'avg': ['average', 'mean', 'avg'],
    'max': ['maximum', 'highest', 'largest', 'max'],
    'min': ['minimum', 'lowest', 'smallest', 'min'],
    'group': ['group by', 'grouped by', 'for each', 'by'],
    'order': ['sort', 'order', 'arrange', 'sorted by'],
    'join': ['join', 'combine', 'merge', 'together with'],
    'insert': ['add', 'insert', 'create', 'new'],
    'update': ['update', 'modify', 'change', 'set'],
    'delete': ['delete', 'remove', 'drop']
}

# Common comparison operators
OPERATORS = {
    'equals': ['=', 'is', 'equals', 'equal to'],
    'greater': ['>', 'greater than', 'more than', 'above'],
    'less': ['<', 'less than', 'below', 'under'],
    'like': ['like', 'contains', 'includes', 'similar to'],
    'in': ['in', 'among', 'one of'],
    'between': ['between', 'from', 'range']
}

class NLToSQLConverter:
    def __init__(self, schema: Dict[str, Any] = None):
        self.schema = schema or {}
        self.tables = list(self.schema.keys()) if self.schema else []
    
    def load_schema(self, schema_file: str):
        """Load database schema from file"""
        with open(schema_file, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        self.tables = list(self.schema.keys())
    
    def extract_table_name(self, query: str) -> str:
        """Extract table name from natural language query"""
        query_lower = query.lower()
        
        # Look for table names in the query
        for table in self.tables:
            if table.lower() in query_lower:
                return table
        
        # Look for common table indicators
        table_indicators = ['from', 'in', 'table', 'data']
        words = query_lower.split()
        
        for i, word in enumerate(words):
            if word in table_indicators and i + 1 < len(words):
                potential_table = words[i + 1]
                # Check if it matches any table (fuzzy)
                for table in self.tables:
                    if potential_table in table.lower() or table.lower() in potential_table:
                        return table
        
        # Default to first table if available
        return self.tables[0] if self.tables else 'table_name'
    
    def extract_columns(self, query: str, table: str) -> List[str]:
        """Extract column names from query"""
        query_lower = query.lower()
        
        if not self.schema or table not in self.schema:
            return ['*']
        
        table_columns = self.schema[table]['columns']
        mentioned_columns = []
        
        # Check for specific column mentions
        for column in table_columns:
            column_name = column['name'].lower()
            if column_name in query_lower:
                mentioned_columns.append(column['name'])
        
        # Check for common column patterns
        if 'name' in query_lower and not mentioned_columns:
            name_cols = [col['name'] for col in table_columns 
                        if 'name' in col['name'].lower()]
            mentioned_columns.extend(name_cols)
        
        if 'id' in query_lower and not mentioned_columns:
            id_cols = [col['name'] for col in table_columns 
                      if 'id' in col['name'].lower()]
            mentioned_columns.extend(id_cols)
        
        return mentioned_columns if mentioned_columns else ['*']
    
    def extract_conditions(self, query: str, table: str) -> str:
        """Extract WHERE conditions from query"""
        query_lower = query.lower()
        
        # Look for condition patterns
        condition_patterns = [
            r'where\s+(.*?)(?:\s+order|\s+group|$)',
            r'with\s+(.*?)(?:\s+order|\s+group|$)',
            r'that\s+(.*?)(?:\s+order|\s+group|$)',
            r'having\s+(.*?)(?:\s+order|\s+group|$)'
        ]
        
        for pattern in condition_patterns:
            match = re.search(pattern, query_lower)
            if match:
                condition_text = match.group(1).strip()
                return self.parse_condition(condition_text, table)
        
        # Look for implicit conditions
        numbers = re.findall(r'\b\d+\b', query)
        strings = re.findall(r"'([^']*)'|\"([^\"]*)\"", query)
        
        if numbers or strings:
            # Try to build a condition
            if table in self.schema:
                columns = self.schema[table]['columns']
                for column in columns:
                    col_name = column['name'].lower()
                    if col_name in query_lower:
                        if numbers and column['type'] in ['int', 'float', 'number']:
                            return f"{column['name']} = {numbers[0]}"
                        elif strings and column['type'] in ['string', 'text', 'varchar']:
                            string_val = strings[0][0] or strings[0][1]
                            return f"{column['name']} = '{string_val}'"
        
        return ""
    
    def parse_condition(self, condition_text: str, table: str) -> str:
        """Parse condition text into SQL WHERE clause"""
        condition_text = condition_text.strip()
        
        # Simple condition parsing
        for op_type, op_variants in OPERATORS.items():
            for op in op_variants:
                if op in condition_text:
                    parts = condition_text.split(op, 1)
                    if len(parts) == 2:
                        left = parts[0].strip()
                        right = parts[1].strip()
                        
                        # Map to SQL operator
                        if op_type == 'equals':
                            return f"{left} = {right}"
                        elif op_type == 'greater':
                            return f"{left} > {right}"
                        elif op_type == 'less':
                            return f"{left} < {right}"
                        elif op_type == 'like':
                            return f"{left} LIKE '%{right}%'"
                        elif op_type == 'in':
                            return f"{left} IN ({right})"
        
        return condition_text
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of SQL query needed"""
        query_lower = query.lower()
        
        # Check for different operation types
        for operation, keywords in SQL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if operation == 'count':
                        return 'count'
                    elif operation in ['sum', 'avg', 'max', 'min']:
                        return 'aggregate'
                    elif operation == 'group':
                        return 'group_by'
                    elif operation == 'order':
                        return 'order_by'
                    elif operation == 'join':
                        return 'join'
                    elif operation == 'insert':
                        return 'insert'
                    elif operation == 'update':
                        return 'update'
                    elif operation == 'delete':
                        return 'delete'
        
        # Default to select
        return 'select'
    
    def convert_to_sql(self, query: str) -> Dict[str, Any]:
        """Convert natural language query to SQL"""
        
        # Extract components
        table = self.extract_table_name(query)
        columns = self.extract_columns(query, table)
        conditions = self.extract_conditions(query, table)
        query_type = self.detect_query_type(query)
        
        # Build SQL based on query type
        if query_type == 'count':
            if conditions:
                sql = SQL_TEMPLATES['count_where'].format(
                    table=table, condition=conditions
                )
            else:
                sql = SQL_TEMPLATES['count'].format(table=table)
        
        elif query_type == 'aggregate':
            # Detect aggregate function
            query_lower = query.lower()
            agg_func = 'COUNT'
            if 'sum' in query_lower:
                agg_func = 'SUM'
            elif 'avg' in query_lower or 'average' in query_lower:
                agg_func = 'AVG'
            elif 'max' in query_lower:
                agg_func = 'MAX'
            elif 'min' in query_lower:
                agg_func = 'MIN'
            
            column = columns[0] if columns != ['*'] else 'column_name'
            sql = SQL_TEMPLATES['aggregate'].format(
                agg_function=agg_func, column=column, table=table
            )
        
        elif query_type == 'select':
            columns_str = ', '.join(columns)
            if conditions:
                sql = SQL_TEMPLATES['select_where'].format(
                    columns=columns_str, table=table, condition=conditions
                )
            else:
                if columns == ['*']:
                    sql = SQL_TEMPLATES['select_all'].format(table=table)
                else:
                    sql = SQL_TEMPLATES['select_columns'].format(
                        columns=columns_str, table=table
                    )
        
        else:
            # Fallback to basic select
            sql = SQL_TEMPLATES['select_all'].format(table=table)
        
        return {
            'sql': sql,
            'query_type': query_type,
            'table': table,
            'columns': columns,
            'conditions': conditions,
            'confidence': self.calculate_confidence(query, sql)
        }
    
    def calculate_confidence(self, nl_query: str, sql_query: str) -> float:
        """Calculate confidence score for the conversion"""
        
        # Factors that increase confidence
        confidence = 0.5  # Base confidence
        
        # Table name found in schema
        if self.schema:
            confidence += 0.2
        
        # Specific columns mentioned
        if 'SELECT *' not in sql_query:
            confidence += 0.1
        
        # Conditions present
        if 'WHERE' in sql_query:
            confidence += 0.1
        
        # Keywords match
        nl_lower = nl_query.lower()
        sql_lower = sql_query.lower()
        
        keyword_matches = 0
        for sql_keyword in ['select', 'where', 'count', 'sum', 'avg', 'max', 'min']:
            if sql_keyword in sql_lower:
                for nl_keyword in SQL_KEYWORDS.get(sql_keyword, []):
                    if nl_keyword in nl_lower:
                        keyword_matches += 1
                        break
        
        confidence += min(0.1, keyword_matches * 0.05)
        
        return min(1.0, confidence)

def create_sample_schema() -> Dict[str, Any]:
    """Create a sample database schema"""
    return {
        'users': {
            'columns': [
                {'name': 'id', 'type': 'int', 'primary_key': True},
                {'name': 'name', 'type': 'string'},
                {'name': 'email', 'type': 'string'},
                {'name': 'age', 'type': 'int'},
                {'name': 'city', 'type': 'string'}
            ]
        },
        'orders': {
            'columns': [
                {'name': 'id', 'type': 'int', 'primary_key': True},
                {'name': 'user_id', 'type': 'int'},
                {'name': 'product', 'type': 'string'},
                {'name': 'amount', 'type': 'float'},
                {'name': 'date', 'type': 'date'}
            ]
        },
        'products': {
            'columns': [
                {'name': 'id', 'type': 'int', 'primary_key': True},
                {'name': 'name', 'type': 'string'},
                {'name': 'price', 'type': 'float'},
                {'name': 'category', 'type': 'string'}
            ]
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Convert natural language to SQL")
    parser.add_argument('query', nargs='?', help='Natural language query to convert')
    parser.add_argument('--schema', help='JSON file with database schema')
    parser.add_argument('--input', help='File with queries to convert')
    parser.add_argument('--output', help='Output file for SQL queries')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--sample-schema', action='store_true', 
                       help='Use sample schema for testing')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = NLToSQLConverter()
    
    # Load schema
    if args.sample_schema:
        converter.schema = create_sample_schema()
        converter.tables = list(converter.schema.keys())
        print("Using sample schema with tables: users, orders, products")
    elif args.schema:
        converter.load_schema(args.schema)
        print(f"Loaded schema with tables: {', '.join(converter.tables)}")
    
    if args.interactive:
        print("\nNatural Language to SQL Converter")
        print("Enter 'quit' to exit")
        
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                result = converter.convert_to_sql(query)
                
                print(f"\nSQL: {result['sql']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Query type: {result['query_type']}")
                
                if result['confidence'] < 0.7:
                    print("⚠️  Low confidence - please verify the SQL")
                
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
    
    elif args.input:
        print(f"Converting queries from {args.input}...")
        
        with open(args.input, 'r', encoding='utf-8') as f:
            if args.input.endswith('.jsonl'):
                queries = [json.loads(line) for line in f]
            else:
                data = json.load(f)
                queries = data if isinstance(data, list) else [data]
        
        results = []
        for item in queries:
            query_text = item.get('query', str(item))
            result = converter.convert_to_sql(query_text)
            result['original_query'] = query_text
            results.append(result)
        
        print(f"Converted {len(results)} queries")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        else:
            for i, result in enumerate(results[:5]):
                print(f"\n{i+1}. {result['original_query']}")
                print(f"   SQL: {result['sql']}")
                print(f"   Confidence: {result['confidence']:.2f}")
    
    elif args.query:
        result = converter.convert_to_sql(args.query)
        
        print(f"Query: {args.query}")
        print(f"SQL: {result['sql']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Query type: {result['query_type']}")
        
        if result['confidence'] < 0.7:
            print("⚠️  Low confidence - please verify the SQL")
    
    else:
        # Show examples
        print("Natural Language to SQL Converter")
        print("\nExample queries:")
        examples = [
            "Show all users",
            "Count the number of orders",
            "Find users with age greater than 25",
            "Get the average order amount",
            "Show users from New York"
        ]
        
        if args.sample_schema:
            converter.schema = create_sample_schema()
            converter.tables = list(converter.schema.keys())
        
        for query in examples:
            result = converter.convert_to_sql(query)
            print(f"\n'{query}' -> {result['sql']}")

if __name__ == "__main__":
    main()