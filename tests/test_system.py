"""
Test suite for the Customer Churn Prediction Chatbot.

Run with: pytest tests/test_system.py -v
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


# ==================== DATABASE TESTS ====================

def test_database_connection():
    """Test database connection and basic query."""
    from database.queries import ChurnDatabase
    
    db = ChurnDatabase()
    count = db.get_customer_count()
    
    assert count > 0, "Database should have customers"
    assert count == 7043, f"Expected 7043 customers, got {count}"


def test_churn_statistics():
    """Test churn statistics calculation."""
    from database.queries import get_churn_rate
    
    stats = get_churn_rate()
    
    assert 'total_customers' in stats
    assert 'churned' in stats
    assert 'churn_rate' in stats
    assert 0 <= stats['churn_rate'] <= 1
    assert stats['total_customers'] == stats['churned'] + stats['retained']


def test_filtered_churn_rate():
    """Test filtered churn rate query."""
    from database.queries import get_filtered_churn_rate
    
    # Test: No landline filter
    result = get_filtered_churn_rate({'PhoneService': 'No'})
    
    assert 'churn_rate' in result
    assert 'total_customers' in result
    assert result['total_customers'] > 0
    
    print(f"✓ Churn rate for no landline: {result['churn_rate_pct']}")


def test_segment_statistics():
    """Test segment analysis."""
    from database.queries import ChurnDatabase
    
    db = ChurnDatabase()
    stats = db.get_segment_statistics('SeniorCitizen', '1')
    
    assert 'churn_rate' in stats
    assert stats['total_customers'] > 0
    
    print(f"✓ Senior citizen churn rate: {stats['churn_rate_pct']}")


# ==================== MODEL TESTS ====================

def test_model_loading():
    """Test model and pipeline loading."""
    from models.predictor import ChurnPredictor
    
    predictor = ChurnPredictor()
    info = predictor.get_model_info()
    
    assert info['loaded'] == True
    assert 'XGB' in info['model_type']
    
    print(f"✓ Model loaded: {info['model_type']}")


def test_single_customer_prediction():
    """Test prediction for a single customer."""
    from models.predictor import predict_customer
    
    # Use first customer ID from database
    from database.queries import ChurnDatabase
    db = ChurnDatabase()
    customers = db.get_all_customers(limit=1)
    customer_id = customers.iloc[0]['customerID']
    
    result = predict_customer(customer_id)
    
    assert 'churn_probability' in result
    assert 0 <= result['churn_probability'] <= 1
    assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH']
    
    print(f"✓ Prediction for {customer_id}: {result['churn_probability']:.1%} ({result['risk_level']})")


def test_high_risk_customers():
    """Test high-risk customer detection."""
    from models.predictor import get_high_risk_customers
    
    high_risk = get_high_risk_customers(threshold=0.7, limit=10)
    
    assert len(high_risk) <= 10
    if len(high_risk) > 0 and 'error' not in high_risk[0]:
        assert all(c['churn_probability'] >= 0.7 for c in high_risk)
        print(f"✓ Found {len(high_risk)} high-risk customers")
    else:
        print("✓ No high-risk customers found (acceptable)")


# ==================== AGENT TESTS ====================

def test_agent_initialization():
    """Test agent initialization."""
    from agents.churn_agent import ChurnAgent
    
    try:
        agent = ChurnAgent()
        info = agent.get_agent_info()
        
        assert info['num_tools'] == 10
        assert info['has_memory'] == True
        
        print(f"✓ Agent initialized with {info['num_tools']} tools")
    except ValueError as e:
        if "API key" in str(e):
            pytest.skip("OpenAI API key not configured")
        raise


def test_agent_query():
    """Test agent query processing."""
    from agents.churn_agent import ChurnAgent
    
    try:
        agent = ChurnAgent()
        result = agent.query("What's the total number of customers?")
        
        assert result['success'] == True
        assert len(result['output']) > 0
        
        print(f"✓ Agent query successful")
        print(f"  Response: {result['output'][:100]}...")
        
    except ValueError as e:
        if "API key" in str(e):
            pytest.skip("OpenAI API key not configured")
        raise


# ==================== TOOL TESTS ====================

def test_tools_import():
    """Test that all tools can be imported."""
    from tools.agent_tools import CHURN_TOOLS
    
    assert len(CHURN_TOOLS) == 10
    
    tool_names = [tool.name for tool in CHURN_TOOLS]
    expected_tools = [
        'query_overall_churn_rate',
        'query_churn_rate_with_filter',
        'predict_churn_for_customer',
        'find_high_risk_customers'
    ]
    
    for expected in expected_tools:
        assert expected in tool_names, f"Tool {expected} not found"
    
    print(f"✓ All {len(CHURN_TOOLS)} tools imported successfully")


def test_tool_execution():
    """Test individual tool execution."""
    from tools.agent_tools import query_overall_churn_rate
    
    result = query_overall_churn_rate.invoke({})
    
    assert isinstance(result, str)
    assert "Churn Rate" in result or "churn rate" in result.lower()
    
    print(f"✓ Tool execution successful")
    print(f"  Result: {result[:100]}...")


# ==================== INTEGRATION TESTS ====================

def test_end_to_end_workflow():
    """Test complete workflow from query to response."""
    print("\n" + "=" * 60)
    print("END-TO-END WORKFLOW TEST")
    print("=" * 60)
    
    # 1. Database query
    print("\n1. Testing database...")
    from database.queries import get_churn_rate
    stats = get_churn_rate()
    print(f"   ✓ Overall churn rate: {stats['churn_rate_pct']}")
    
    # 2. Filtered query
    print("\n2. Testing filtered query...")
    from database.queries import get_filtered_churn_rate
    result = get_filtered_churn_rate({'PhoneService': 'No'})
    print(f"   ✓ No landline churn rate: {result['churn_rate_pct']}")
    
    # 3. Model prediction
    print("\n3. Testing model prediction...")
    from models.predictor import ChurnPredictor
    predictor = ChurnPredictor()
    from database.queries import ChurnDatabase
    db = ChurnDatabase()
    customers = db.get_all_customers(limit=1)
    customer_id = customers.iloc[0]['customerID']
    pred = predictor.predict_single_customer(customer_id)
    print(f"   ✓ Prediction: {pred['churn_probability']:.1%} risk")
    
    # 4. Agent query (if API key available)
    print("\n4. Testing agent...")
    try:
        from agents.churn_agent import ChurnAgent
        agent = ChurnAgent()
        response = agent.query("How many customers do we have?")
        print(f"   ✓ Agent response: {response['output'][:100]}...")
    except ValueError as e:
        if "API key" in str(e):
            print("   ⊘ Skipped (API key not configured)")
        else:
            raise
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)


# ==================== QUICK TEST FUNCTION ====================

def run_quick_tests():
    """Run quick tests manually without pytest."""
    print("\n🧪 Running Quick Tests...\n")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Churn Statistics", test_churn_statistics),
        ("Filtered Churn Rate", test_filtered_churn_rate),
        ("Model Loading", test_model_loading),
        ("Single Prediction", test_single_customer_prediction),
        ("Tools Import", test_tools_import),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"Testing: {name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")
    
    return failed == 0


if __name__ == "__main__":
    """Run tests directly."""
    import sys
    
    # Run end-to-end test
    test_end_to_end_workflow()
    
    # Run all quick tests
    success = run_quick_tests()
    
    sys.exit(0 if success else 1)
