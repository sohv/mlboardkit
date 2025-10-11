#!/usr/bin/env python3
"""
Database Utilities for ML/AI Data Operations

Common database operations for ML workflows including querying, data export,
schema management, and ETL operations with support for multiple database types.

Usage:
    python3 db_utilities.py connect --db-type sqlite --path data.db
    python3 db_utilities.py query --sql "SELECT * FROM users LIMIT 10" --output results.csv
    python3 db_utilities.py export --table users --format json --output users.json
    python3 db_utilities.py schema --action describe --table users
    python3 db_utilities.py etl --source source.db --target target.db --table users
"""

import argparse
import sqlite3
import pandas as pd
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import csv
from urllib.parse import urlparse
import tempfile


class DatabaseConnection:
    """Database connection manager supporting multiple database types."""
    
    def __init__(self, db_type: str, connection_params: Dict[str, Any]):
        self.db_type = db_type.lower()
        self.connection_params = connection_params
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        try:
            if self.db_type == 'sqlite':
                import sqlite3
                db_path = self.connection_params.get('path', ':memory:')
                self.connection = sqlite3.connect(db_path)
                self.connection.row_factory = sqlite3.Row  # Enable dict-like access
                
            elif self.db_type == 'mysql':
                try:
                    import mysql.connector
                    self.connection = mysql.connector.connect(**self.connection_params)
                except ImportError:
                    raise ImportError("mysql-connector-python required for MySQL: pip install mysql-connector-python")
                
            elif self.db_type == 'postgresql':
                try:
                    import psycopg2
                    import psycopg2.extras
                    self.connection = psycopg2.connect(**self.connection_params)
                    # Enable dict-like access
                    self.connection.cursor_factory = psycopg2.extras.RealDictCursor
                except ImportError:
                    raise ImportError("psycopg2 required for PostgreSQL: pip install psycopg2-binary")
                
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
            print(f"✅ Connected to {self.db_type} database")
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute query and return results."""
        cursor = self.connection.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Handle different cursor types
            if hasattr(cursor, 'fetchall'):
                if self.db_type == 'sqlite':
                    results = [dict(row) for row in cursor.fetchall()]
                else:
                    results = cursor.fetchall()
                    if results and not isinstance(results[0], dict):
                        # Convert to dict if needed
                        columns = [desc[0] for desc in cursor.description]
                        results = [dict(zip(columns, row)) for row in results]
            else:
                results = []
            
            self.connection.commit()
            return results
            
        except Exception as e:
            self.connection.rollback()
            print(f"❌ Query execution failed: {e}")
            print(f"Query: {query}")
            raise
        finally:
            cursor.close()
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information."""
        if self.db_type == 'sqlite':
            query = f"PRAGMA table_info({table_name})"
        elif self.db_type == 'mysql':
            query = f"DESCRIBE {table_name}"
        elif self.db_type == 'postgresql':
            query = f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """
        else:
            raise ValueError(f"Schema inspection not supported for {self.db_type}")
        
        results = self.execute_query(query)
        return {
            'table_name': table_name,
            'columns': results,
            'row_count': self.get_row_count(table_name)
        }
    
    def get_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(query)
        return result[0]['count'] if result else 0
    
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        if self.db_type == 'sqlite':
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        elif self.db_type == 'mysql':
            query = "SHOW TABLES"
        elif self.db_type == 'postgresql':
            query = """
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
        else:
            raise ValueError(f"Table listing not supported for {self.db_type}")
        
        results = self.execute_query(query)
        
        # Extract table names from results
        if self.db_type == 'sqlite':
            return [row['name'] for row in results]
        elif self.db_type == 'mysql':
            # MySQL returns results with table name as key
            return [list(row.values())[0] for row in results]
        elif self.db_type == 'postgresql':
            return [row['table_name'] for row in results]
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("🔌 Database connection closed")


class DatabaseUtilities:
    """Main database utilities class."""
    
    def __init__(self, connection: DatabaseConnection):
        self.db = connection
    
    def export_table(self, table_name: str, output_path: str, 
                    format: str = 'csv', limit: Optional[int] = None,
                    where_clause: Optional[str] = None) -> Path:
        """Export table data to file."""
        print(f"📤 Exporting table '{table_name}' to {format.upper()}")
        
        # Build query
        query = f"SELECT * FROM {table_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute query
        data = self.db.execute_query(query)
        
        if not data:
            print("⚠️  No data found")
            return Path(output_path)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format.lower() == 'csv':
            self._export_to_csv(data, output_path)
        elif format.lower() == 'json':
            self._export_to_json(data, output_path)
        elif format.lower() == 'excel':
            self._export_to_excel(data, output_path)
        elif format.lower() == 'parquet':
            self._export_to_parquet(data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"✅ Exported {len(data)} rows to {output_path}")
        return output_path
    
    def _export_to_csv(self, data: List[Dict], output_path: Path):
        """Export data to CSV."""
        if not data:
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    
    def _export_to_json(self, data: List[Dict], output_path: Path):
        """Export data to JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    
    def _export_to_excel(self, data: List[Dict], output_path: Path):
        """Export data to Excel."""
        try:
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel export: pip install pandas openpyxl")
    
    def _export_to_parquet(self, data: List[Dict], output_path: Path):
        """Export data to Parquet."""
        try:
            df = pd.DataFrame(data)
            df.to_parquet(output_path, index=False)
        except ImportError:
            raise ImportError("pandas and pyarrow required for Parquet export: pip install pandas pyarrow")
    
    def import_data(self, table_name: str, data_path: str, 
                   if_exists: str = 'append', create_table: bool = True) -> int:
        """Import data from file to database table."""
        print(f"📥 Importing data to table '{table_name}'")
        
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data based on file extension
        suffix = data_path.suffix.lower()
        
        if suffix == '.csv':
            df = pd.read_csv(data_path)
        elif suffix == '.json':
            df = pd.read_json(data_path)
        elif suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        elif suffix == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        print(f"📊 Loaded {len(df)} rows from {data_path}")
        
        # Convert DataFrame to database
        row_count = self._dataframe_to_db(df, table_name, if_exists, create_table)
        
        print(f"✅ Imported {row_count} rows to '{table_name}'")
        return row_count
    
    def _dataframe_to_db(self, df: pd.DataFrame, table_name: str, 
                        if_exists: str, create_table: bool) -> int:
        """Convert DataFrame to database table."""
        if self.db.db_type == 'sqlite':
            # Use pandas to_sql for SQLite
            df.to_sql(table_name, self.db.connection, if_exists=if_exists, index=False)
            return len(df)
        
        else:
            # Manual insertion for other databases
            if create_table and if_exists == 'replace':
                self._create_table_from_dataframe(df, table_name)
            
            # Insert data
            columns = list(df.columns)
            placeholders = ', '.join(['%s'] * len(columns))
            
            if self.db.db_type == 'sqlite':
                placeholders = ', '.join(['?'] * len(columns))
            
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # Insert in batches
            batch_size = 1000
            rows_inserted = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                values = [tuple(row) for row in batch.values]
                
                cursor = self.db.connection.cursor()
                try:
                    if self.db.db_type == 'mysql':
                        cursor.executemany(query, values)
                    else:
                        for value in values:
                            cursor.execute(query, value)
                    
                    self.db.connection.commit()
                    rows_inserted += len(values)
                    
                except Exception as e:
                    self.db.connection.rollback()
                    print(f"❌ Batch insertion failed: {e}")
                    raise
                finally:
                    cursor.close()
            
            return rows_inserted
    
    def _create_table_from_dataframe(self, df: pd.DataFrame, table_name: str):
        """Create table based on DataFrame schema."""
        # Map pandas dtypes to SQL types
        type_mapping = {
            'int64': 'INTEGER',
            'float64': 'REAL',
            'object': 'TEXT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP'
        }
        
        columns = []
        for col, dtype in df.dtypes.items():
            sql_type = type_mapping.get(str(dtype), 'TEXT')
            columns.append(f"{col} {sql_type}")
        
        # Drop table if exists
        drop_query = f"DROP TABLE IF EXISTS {table_name}"
        self.db.execute_query(drop_query)
        
        # Create table
        create_query = f"CREATE TABLE {table_name} ({', '.join(columns)})"
        self.db.execute_query(create_query)
        
        print(f"✅ Created table '{table_name}' with {len(columns)} columns")
    
    def execute_sql_file(self, sql_file_path: str) -> List[Dict[str, Any]]:
        """Execute SQL commands from file."""
        sql_file_path = Path(sql_file_path)
        if not sql_file_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_file_path}")
        
        print(f"📜 Executing SQL file: {sql_file_path}")
        
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Split into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        results = []
        for i, statement in enumerate(statements):
            print(f"📝 Executing statement {i + 1}/{len(statements)}")
            try:
                result = self.db.execute_query(statement)
                results.extend(result)
            except Exception as e:
                print(f"❌ Statement {i + 1} failed: {e}")
                print(f"Statement: {statement[:100]}...")
                raise
        
        print(f"✅ Executed {len(statements)} SQL statements")
        return results
    
    def create_database_backup(self, backup_path: str, tables: Optional[List[str]] = None) -> Path:
        """Create database backup."""
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Creating database backup: {backup_path}")
        
        if not tables:
            tables = self.db.list_tables()
        
        backup_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'database_type': self.db.db_type,
                'tables': tables
            },
            'data': {}
        }
        
        for table in tables:
            print(f"📊 Backing up table: {table}")
            try:
                table_data = self.db.execute_query(f"SELECT * FROM {table}")
                backup_data['data'][table] = table_data
                print(f"✅ Backed up {len(table_data)} rows from {table}")
            except Exception as e:
                print(f"⚠️  Failed to backup table {table}: {e}")
        
        # Save backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"✅ Database backup created: {backup_path}")
        return backup_path
    
    def restore_database_backup(self, backup_path: str, 
                               tables: Optional[List[str]] = None) -> Dict[str, int]:
        """Restore database from backup."""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        print(f"🔄 Restoring database from: {backup_path}")
        
        with open(backup_path, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        restored_tables = {}
        
        for table_name, table_data in backup_data['data'].items():
            if tables is None or table_name in tables:
                print(f"📥 Restoring table: {table_name}")
                try:
                    # Create DataFrame and import
                    df = pd.DataFrame(table_data)
                    if not df.empty:
                        row_count = self._dataframe_to_db(df, table_name, 'replace', True)
                        restored_tables[table_name] = row_count
                        print(f"✅ Restored {row_count} rows to {table_name}")
                except Exception as e:
                    print(f"❌ Failed to restore table {table_name}: {e}")
        
        print(f"✅ Database restore completed: {len(restored_tables)} tables")
        return restored_tables
    
    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """Analyze table and provide statistics."""
        print(f"📊 Analyzing table: {table_name}")
        
        # Get basic info
        table_info = self.db.get_table_info(table_name)
        
        # Get sample data
        sample_data = self.db.execute_query(f"SELECT * FROM {table_name} LIMIT 5")
        
        # Get column statistics for numeric columns
        analysis = {
            'table_info': table_info,
            'sample_data': sample_data,
            'column_stats': {}
        }
        
        # Analyze each column
        for column in table_info['columns']:
            col_name = column.get('name') or column.get('column_name') or column.get('Field')
            if col_name:
                stats = self._analyze_column(table_name, col_name)
                analysis['column_stats'][col_name] = stats
        
        return analysis
    
    def _analyze_column(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Analyze individual column."""
        stats = {}
        
        try:
            # Null count
            null_query = f"SELECT COUNT(*) as null_count FROM {table_name} WHERE {column_name} IS NULL"
            null_result = self.db.execute_query(null_query)
            stats['null_count'] = null_result[0]['null_count']
            
            # Distinct count
            distinct_query = f"SELECT COUNT(DISTINCT {column_name}) as distinct_count FROM {table_name}"
            distinct_result = self.db.execute_query(distinct_query)
            stats['distinct_count'] = distinct_result[0]['distinct_count']
            
            # Try numeric statistics
            try:
                numeric_query = f"""
                    SELECT 
                        MIN({column_name}) as min_val,
                        MAX({column_name}) as max_val,
                        AVG({column_name}) as avg_val
                    FROM {table_name} 
                    WHERE {column_name} IS NOT NULL
                """
                numeric_result = self.db.execute_query(numeric_query)
                if numeric_result:
                    stats.update(numeric_result[0])
            except:
                # Not a numeric column
                pass
                
        except Exception as e:
            stats['error'] = str(e)
        
        return stats


def parse_connection_string(connection_string: str) -> Tuple[str, Dict[str, Any]]:
    """Parse database connection string."""
    parsed = urlparse(connection_string)
    
    db_type = parsed.scheme
    
    if db_type == 'sqlite':
        return 'sqlite', {'path': parsed.path}
    
    elif db_type in ['mysql', 'postgresql']:
        params = {
            'host': parsed.hostname,
            'port': parsed.port,
            'database': parsed.path.lstrip('/'),
            'user': parsed.username,
            'password': parsed.password
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return db_type, params
    
    else:
        raise ValueError(f"Unsupported database type in connection string: {db_type}")


def main():
    parser = argparse.ArgumentParser(description="Database utilities for ML/AI workflows")
    parser.add_argument('--config', help='Configuration file (YAML/JSON)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Connect command
    connect_parser = subparsers.add_parser('connect', help='Test database connection')
    connect_parser.add_argument('--db-type', choices=['sqlite', 'mysql', 'postgresql'], 
                               required=True, help='Database type')
    connect_parser.add_argument('--connection-string', help='Database connection string')
    connect_parser.add_argument('--path', help='Database file path (SQLite)')
    connect_parser.add_argument('--host', help='Database host')
    connect_parser.add_argument('--port', type=int, help='Database port')
    connect_parser.add_argument('--database', help='Database name')
    connect_parser.add_argument('--user', help='Username')
    connect_parser.add_argument('--password', help='Password')

    # Query command
    query_parser = subparsers.add_parser('query', help='Execute SQL query')
    query_parser.add_argument('--sql', help='SQL query to execute')
    query_parser.add_argument('--sql-file', help='SQL file to execute')
    query_parser.add_argument('--output', help='Output file path')
    query_parser.add_argument('--format', choices=['csv', 'json', 'excel'], 
                             default='csv', help='Output format')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export table data')
    export_parser.add_argument('--table', required=True, help='Table name to export')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['csv', 'json', 'excel', 'parquet'], 
                              default='csv', help='Export format')
    export_parser.add_argument('--limit', type=int, help='Limit number of rows')
    export_parser.add_argument('--where', help='WHERE clause')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import data to table')
    import_parser.add_argument('--table', required=True, help='Target table name')
    import_parser.add_argument('--data', required=True, help='Data file path')
    import_parser.add_argument('--if-exists', choices=['append', 'replace'], 
                              default='append', help='Action if table exists')

    # Schema command
    schema_parser = subparsers.add_parser('schema', help='Schema operations')
    schema_parser.add_argument('--action', choices=['list', 'describe', 'analyze'], 
                              required=True, help='Schema action')
    schema_parser.add_argument('--table', help='Table name (for describe/analyze)')

    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument('--output', required=True, help='Backup file path')
    backup_parser.add_argument('--tables', nargs='*', help='Specific tables to backup')

    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore database backup')
    restore_parser.add_argument('--backup', required=True, help='Backup file path')
    restore_parser.add_argument('--tables', nargs='*', help='Specific tables to restore')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # Load configuration
        config = {}
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    if config_path.suffix.lower() in ['.yaml', '.yml']:
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)

        # Determine connection parameters
        if hasattr(args, 'connection_string') and args.connection_string:
            db_type, conn_params = parse_connection_string(args.connection_string)
        elif hasattr(args, 'db_type') and args.db_type:
            db_type = args.db_type
            conn_params = {}
            
            if db_type == 'sqlite':
                conn_params['path'] = args.path or config.get('path', ':memory:')
            else:
                conn_params.update({
                    'host': args.host or config.get('host', 'localhost'),
                    'port': args.port or config.get('port'),
                    'database': args.database or config.get('database'),
                    'user': args.user or config.get('user'),
                    'password': args.password or config.get('password')
                })
        else:
            # Try to get from config
            db_type = config.get('db_type', 'sqlite')
            conn_params = config.get('connection', {})

        # Create connection
        db_connection = DatabaseConnection(db_type, conn_params)
        db_utils = DatabaseUtilities(db_connection)

        # Execute commands
        if args.command == 'connect':
            print("🔌 Connection test successful!")
            tables = db_connection.list_tables()
            print(f"📋 Found {len(tables)} tables: {', '.join(tables[:5])}")
            if len(tables) > 5:
                print(f"    ... and {len(tables) - 5} more")

        elif args.command == 'query':
            if args.sql:
                results = db_connection.execute_query(args.sql)
            elif args.sql_file:
                results = db_utils.execute_sql_file(args.sql_file)
            else:
                print("❌ Either --sql or --sql-file must be provided")
                return

            print(f"📊 Query returned {len(results)} rows")
            
            if args.output and results:
                output_path = Path(args.output)
                if args.format == 'csv':
                    db_utils._export_to_csv(results, output_path)
                elif args.format == 'json':
                    db_utils._export_to_json(results, output_path)
                elif args.format == 'excel':
                    db_utils._export_to_excel(results, output_path)
                print(f"📁 Results saved to: {output_path}")
            elif results:
                # Print first few rows
                for i, row in enumerate(results[:5]):
                    print(f"Row {i + 1}: {row}")
                if len(results) > 5:
                    print(f"... and {len(results) - 5} more rows")

        elif args.command == 'export':
            db_utils.export_table(args.table, args.output, args.format, args.limit, args.where)

        elif args.command == 'import':
            db_utils.import_data(args.table, args.data, args.if_exists)

        elif args.command == 'schema':
            if args.action == 'list':
                tables = db_connection.list_tables()
                print(f"📋 Tables ({len(tables)}):")
                for table in tables:
                    row_count = db_connection.get_row_count(table)
                    print(f"  {table}: {row_count} rows")
            
            elif args.action == 'describe':
                if not args.table:
                    print("❌ --table required for describe action")
                    return
                table_info = db_connection.get_table_info(args.table)
                print(f"📊 Table: {table_info['table_name']}")
                print(f"Rows: {table_info['row_count']}")
                print("Columns:")
                for col in table_info['columns']:
                    print(f"  {col}")
            
            elif args.action == 'analyze':
                if not args.table:
                    print("❌ --table required for analyze action")
                    return
                analysis = db_utils.analyze_table(args.table)
                print(f"📈 Analysis for {args.table}:")
                print(f"Rows: {analysis['table_info']['row_count']}")
                print("Column Statistics:")
                for col_name, stats in analysis['column_stats'].items():
                    print(f"  {col_name}: {stats}")

        elif args.command == 'backup':
            db_utils.create_database_backup(args.output, args.tables)

        elif args.command == 'restore':
            restored = db_utils.restore_database_backup(args.backup, args.tables)
            print(f"📊 Restored tables: {restored}")

        # Close connection
        db_connection.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()