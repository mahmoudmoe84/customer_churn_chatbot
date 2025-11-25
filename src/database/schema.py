"""
Database schema documentation for LLM context.

This file provides schema information that will be included in the
agent's system prompt so it knows what columns exist and can generate
appropriate SQL queries.
"""

# Customer Churn Database Schema
CUSTOMER_CHURN_SCHEMA = """
Table: customer_churn

Columns:
1. customerID (TEXT) - Unique customer identifier
2. gender (TEXT) - Customer gender: 'Female' or 'Male'
3. SeniorCitizen (INTEGER) - Whether customer is senior: 0 (No) or 1 (Yes)
4. Partner (TEXT) - Has partner: 'Yes' or 'No'
5. Dependents (TEXT) - Has dependents: 'Yes' or 'No'
6. tenure (INTEGER) - Months with company (0-72)
7. PhoneService (TEXT) - Has phone service: 'Yes' or 'No'
8. MultipleLines (TEXT) - Multiple phone lines: 'Yes', 'No', or 'No phone service'
9. InternetService (TEXT) - Internet service type: 'DSL', 'Fiber optic', or 'No'
10. OnlineSecurity (TEXT) - Has online security: 'Yes', 'No', or 'No internet service'
11. OnlineBackup (TEXT) - Has online backup: 'Yes', 'No', or 'No internet service'
12. DeviceProtection (TEXT) - Has device protection: 'Yes', 'No', or 'No internet service'
13. TechSupport (TEXT) - Has tech support: 'Yes', 'No', or 'No internet service'
14. StreamingTV (TEXT) - Has streaming TV: 'Yes', 'No', or 'No internet service'
15. StreamingMovies (TEXT) - Has streaming movies: 'Yes', 'No', or 'No internet service'
16. Contract (TEXT) - Contract type: 'Month-to-month', 'One year', or 'Two year'
17. PaperlessBilling (TEXT) - Uses paperless billing: 'Yes' or 'No'
18. PaymentMethod (TEXT) - Payment method: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
19. MonthlyCharges (REAL) - Monthly charge amount ($)
20. TotalCharges (REAL) - Total charges to date ($)
21. Churn (TEXT) - Customer churned: 'Yes' or 'No' (TARGET VARIABLE)

Total Records: ~7,043 customers
"""

# Common query patterns
QUERY_EXAMPLES = """
Common Query Patterns:

1. Overall churn rate:
   - Use: calculate_churn_statistics()

2. Filtered churn rate (e.g., "no landline"):
   - Use: calculate_filtered_churn_rate({'PhoneService': 'No'})

3. Senior citizen churn:
   - Use: get_segment_statistics('SeniorCitizen', '1')

4. Month-to-month contract churn:
   - Use: calculate_filtered_churn_rate({'Contract': 'Month-to-month'})

5. Multiple filters (e.g., "senior citizens with fiber optic"):
   - Use: calculate_filtered_churn_rate({
       'SeniorCitizen': '1',
       'InternetService': 'Fiber optic'
   })

6. Find high-value churned customers:
   - Use: query_with_filters({
       'Churn': 'Yes',
       'Contract': 'Two year'
   }, limit=50)

Important Notes:
- PhoneService='No' means no landline
- InternetService='No' means no internet
- SeniorCitizen: 0=No, 1=Yes (stored as integer)
- All other Yes/No fields are stored as TEXT ('Yes' or 'No')
"""

# Mapping for natural language to database values
VALUE_MAPPINGS = {
    'landline': {
        'has_landline': {'PhoneService': 'Yes'},
        'no_landline': {'PhoneService': 'No'}
    },
    'internet': {
        'has_internet': "InternetService IN ('DSL', 'Fiber optic')",
        'no_internet': {'InternetService': 'No'},
        'fiber': {'InternetService': 'Fiber optic'},
        'dsl': {'InternetService': 'DSL'}
    },
    'senior': {
        'senior': {'SeniorCitizen': '1'},
        'not_senior': {'SeniorCitizen': '0'}
    },
    'contract': {
        'month_to_month': {'Contract': 'Month-to-month'},
        'one_year': {'Contract': 'One year'},
        'two_year': {'Contract': 'Two year'}
    }
}


def get_schema_context() -> str:
    """Get full schema context for LLM prompt."""
    return f"{CUSTOMER_CHURN_SCHEMA}\n\n{QUERY_EXAMPLES}"
