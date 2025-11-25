"""
Database query functions for customer churn analysis.

This module provides:
1. Pre-built safe queries for common operations
2. Dynamic SQL generation with filtering capabilities
3. Churn statistics and analytics functions
"""

import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional, Any
import re
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config import DATABASE_PATH


class ChurnDatabase:
    """Handle all database operations for customer churn data."""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection."""
        self.db_path = db_path or DATABASE_PATH
        self.engine = create_engine(f'sqlite:///{self.db_path}')
    
    def get_table_schema(self) -> Dict[str, List[str]]:
        """Get database schema information for SQL generation."""
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table';
        """
        with self.engine.connect() as conn:
            tables = pd.read_sql(text(query), conn)
        
        schema = {}
        for table in tables['name']:
            query = f"PRAGMA table_info({table})"
            with self.engine.connect() as conn:
                columns = pd.read_sql(text(query), conn)
            schema[table] = columns['name'].tolist()
        
        return schema
    
    def get_column_info(self) -> pd.DataFrame:
        """Get detailed column information for customer_churn table."""
        query = "PRAGMA table_info(customer_churn)"
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)
    
    # ==================== PRE-BUILT SAFE QUERIES ====================
    
    def get_all_customers(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch all customer records.
        
        Args:
            limit: Optional limit on number of records
            
        Returns:
            DataFrame with all customer data
        """
        query = "SELECT * FROM customer_churn"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn)
    
    def get_customer_by_id(self, customer_id: str) -> pd.DataFrame:
        """
        Get specific customer by ID.
        
        Args:
            customer_id: Customer ID to lookup
            
        Returns:
            DataFrame with customer data (empty if not found)
        """
        query = "SELECT * FROM customer_churn WHERE customerID = :customer_id"
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params={'customer_id': customer_id})
    
    def calculate_churn_statistics(self) -> Dict[str, Any]:
        """
        Calculate overall churn metrics.
        
        Returns:
            Dict with churn statistics:
            - total_customers: Total number of customers
            - churned: Number of churned customers
            - retained: Number of retained customers
            - churn_rate: Percentage of churned customers
        """
        query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
            SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) as retained
        FROM customer_churn
        """
        
        with self.engine.connect() as conn:
            result = pd.read_sql(text(query), conn).iloc[0]
        
        total = int(result['total'])
        churned = int(result['churned'])
        retained = int(result['retained'])
        churn_rate = churned / total if total > 0 else 0.0
        
        return {
            'total_customers': total,
            'churned': churned,
            'retained': retained,
            'churn_rate': churn_rate,
            'churn_rate_pct': f"{churn_rate * 100:.2f}%"
        }
    
    def get_segment_statistics(self, segment_column: str, segment_value: str) -> Dict[str, Any]:
        """
        Calculate churn statistics for a specific customer segment.
        
        Args:
            segment_column: Column name to segment by (e.g., 'SeniorCitizen', 'Contract')
            segment_value: Value to filter (e.g., '1', 'Month-to-month')
            
        Returns:
            Dict with segment-specific statistics
        """
        # Sanitize column name to prevent SQL injection
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', segment_column):
            raise ValueError(f"Invalid column name: {segment_column}")
        
        query = f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
            SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) as retained
        FROM customer_churn
        WHERE {segment_column} = :segment_value
        """
        
        with self.engine.connect() as conn:
            result = pd.read_sql(text(query), conn, params={'segment_value': segment_value})
        
        if len(result) == 0 or result.iloc[0]['total'] == 0:
            return {
                'segment': f"{segment_column}={segment_value}",
                'total_customers': 0,
                'churned': 0,
                'retained': 0,
                'churn_rate': 0.0,
                'error': 'No data found for this segment'
            }
        
        row = result.iloc[0]
        total = int(row['total'])
        churned = int(row['churned'])
        retained = int(row['retained'])
        churn_rate = churned / total if total > 0 else 0.0
        
        return {
            'segment': f"{segment_column}={segment_value}",
            'total_customers': total,
            'churned': churned,
            'retained': retained,
            'churn_rate': churn_rate,
            'churn_rate_pct': f"{churn_rate * 100:.2f}%"
        }
    
    # ==================== DYNAMIC SQL GENERATION ====================
    
    def query_with_filters(self, filters: Dict[str, Any], limit: Optional[int] = 100) -> pd.DataFrame:
        """
        Query customers with dynamic filters.
        
        Example:
            filters = {
                'PhoneService': 'No',
                'InternetService': 'Fiber optic',
                'Churn': 'Yes'
            }
        
        Args:
            filters: Dictionary of column:value pairs to filter
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with filtered customer data
        """
        # Start with base query
        query = "SELECT * FROM customer_churn WHERE 1=1"
        params = {}
        
        # Add filters dynamically
        for idx, (column, value) in enumerate(filters.items()):
            # Validate column name (prevent SQL injection)
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column):
                raise ValueError(f"Invalid column name: {column}")
            
            param_name = f"param_{idx}"
            query += f" AND {column} = :{param_name}"
            params[param_name] = value
        
        if limit:
            query += f" LIMIT {int(limit)}"
        
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    
    def calculate_filtered_churn_rate(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate churn rate for filtered subset of customers.
        
        Example for "churn rate of people with no landline":
            filters = {'PhoneService': 'No'}
        
        Args:
            filters: Dictionary of column:value pairs to filter
            
        Returns:
            Dict with filtered churn statistics
        """
        # Build WHERE clause
        where_clauses = []
        params = {}
        
        for idx, (column, value) in enumerate(filters.items()):
            # Validate column name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column):
                raise ValueError(f"Invalid column name: {column}")
            
            param_name = f"param_{idx}"
            where_clauses.append(f"{column} = :{param_name}")
            params[param_name] = value
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
            SUM(CASE WHEN Churn = 'No' THEN 1 ELSE 0 END) as retained
        FROM customer_churn
        WHERE {where_clause}
        """
        
        with self.engine.connect() as conn:
            result = pd.read_sql(text(query), conn, params=params)
        
        if len(result) == 0 or result.iloc[0]['total'] == 0:
            return {
                'filters': filters,
                'total_customers': 0,
                'churned': 0,
                'retained': 0,
                'churn_rate': 0.0,
                'error': 'No data found matching filters'
            }
        
        row = result.iloc[0]
        total = int(row['total'])
        churned = int(row['churned'])
        retained = int(row['retained'])
        churn_rate = churned / total if total > 0 else 0.0
        
        # Create human-readable filter description
        filter_desc = ", ".join([f"{k}={v}" for k, v in filters.items()])
        
        return {
            'filters': filters,
            'filter_description': filter_desc,
            'total_customers': total,
            'churned': churned,
            'retained': retained,
            'churn_rate': churn_rate,
            'churn_rate_pct': f"{churn_rate * 100:.2f}%"
        }
    
    def execute_safe_query(self, sql_query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Execute a parameterized SQL query safely.
        
        IMPORTANT: Only SELECT queries are allowed for safety.
        
        Args:
            sql_query: SQL query with :parameter placeholders
            params: Dictionary of parameter values
            
        Returns:
            DataFrame with query results
            
        Raises:
            ValueError: If query contains non-SELECT operations
        """
        # Security check: Only allow SELECT queries
        query_upper = sql_query.strip().upper()
        if not query_upper.startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed")
        
        # Block dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'EXEC']
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                raise ValueError(f"Query contains forbidden keyword: {keyword}")
        
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql_query), conn, params=params or {})
    
    # ==================== ANALYTICS FUNCTIONS ====================
    
    def get_top_churn_factors(self, top_n: int = 5) -> pd.DataFrame:
        """
        Analyze which factors have highest correlation with churn.
        
        Args:
            top_n: Number of top factors to return
            
        Returns:
            DataFrame with factors and their churn rates
        """
        # This would be more sophisticated with feature importance from the model
        # For now, we'll analyze categorical variables
        
        categorical_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
        
        results = []
        
        for col in categorical_cols:
            query = f"""
            SELECT 
                {col} as category,
                COUNT(*) as total,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
                CAST(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as churn_rate
            FROM customer_churn
            GROUP BY {col}
            ORDER BY churn_rate DESC
            LIMIT 1
            """
            
            with self.engine.connect() as conn:
                result = pd.read_sql(text(query), conn)
            
            if len(result) > 0:
                row = result.iloc[0]
                results.append({
                    'factor': col,
                    'value': row['category'],
                    'churn_rate': row['churn_rate'],
                    'sample_size': row['total']
                })
        
        df = pd.DataFrame(results)
        return df.nlargest(top_n, 'churn_rate')
    
    def get_customer_count(self) -> int:
        """Get total number of customers."""
        query = "SELECT COUNT(*) as count FROM customer_churn"
        with self.engine.connect() as conn:
            result = pd.read_sql(text(query), conn)
        return int(result.iloc[0]['count'])


# ==================== CONVENIENCE FUNCTIONS ====================

def get_database() -> ChurnDatabase:
    """Get database instance (singleton pattern)."""
    return ChurnDatabase()


# Quick access functions for common operations
def get_churn_rate() -> Dict[str, Any]:
    """Quick function to get overall churn rate."""
    db = get_database()
    return db.calculate_churn_statistics()


def get_filtered_churn_rate(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick function to get churn rate with filters.
    
    Example for "people with no landline":
        get_filtered_churn_rate({'PhoneService': 'No'})
    """
    db = get_database()
    return db.calculate_filtered_churn_rate(filters)


def query_customers(filters: Dict[str, Any], limit: int = 100) -> pd.DataFrame:
    """Quick function to query customers with filters."""
    db = get_database()
    return db.query_with_filters(filters, limit)
