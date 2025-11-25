# Agent Evaluation Tests

Comprehensive test suite for evaluating the churn prediction chatbot agent.

## Features

### 1. **Golden Test Dataset**
Predefined test cases covering:
- Basic queries (overall churn rate)
- Filtered queries (churn by segment)
- Predictions (individual customer risk)
- Comparisons (retained vs high-risk)
- Pattern analysis (deep insights)
- Financial analysis (revenue impact)
- Context awareness (follow-up handling)

### 2. **LLM-as-a-Judge Evaluation**
Uses GPT-4o-mini to evaluate responses on:
- **Accuracy**: No hallucinations, factual data
- **Relevance**: Directly answers the question
- **Completeness**: All expected elements present
- **Professionalism**: Clear, appropriate tone

Scores range from 1-5 with detailed reasoning.

### 3. **Tool Selection Validation**
Verifies the agent:
- Selects the correct tool for each query type
- Completes tasks within iteration limits
- Uses specialized tools (e.g., `compare_retained_vs_high_risk_customers`)

### 4. **Iteration Efficiency Checks**
Monitors:
- Number of tool calls per query
- Ensures comparisons complete in 1-5 iterations
- Flags queries hitting max iteration limit

### 5. **Content Validation**
Checks responses contain:
- Expected keywords/phrases
- Proper formatting (no text concatenation)
- Currency symbols and numbers

## Usage

### Run All Tests
```bash
# Using pytest
pytest tests/test_agent_evaluation.py -v

# Using Python directly
python tests/test_agent_evaluation.py
```

### Run Specific Categories
```bash
# Test only basic queries
pytest tests/test_agent_evaluation.py::test_basic_queries

# Test comparisons
pytest tests/test_agent_evaluation.py::test_comparisons

# Run by category using Python
python tests/test_agent_evaluation.py --category comparison
```

### Run Single Test
```bash
python tests/test_agent_evaluation.py --test-id retained_vs_high_risk_comparison
```

## Test Cases

| ID | Query | Expected Tool | Category |
|----|-------|--------------|----------|
| `basic_churn_rate` | "What is the overall churn rate?" | `query_overall_churn_rate` | basic_query |
| `filtered_churn_rate` | "Churn rate for Month-to-month?" | `query_churn_rate_with_filter` | filtered_query |
| `high_risk_identification` | "Show top 5 high-risk customers" | `find_high_risk_customers` | prediction |
| `retained_vs_high_risk_comparison` | "Compare top 5 retained vs high risk" | `compare_retained_vs_high_risk_customers` | comparison |
| `pattern_analysis` | "Analyze high-risk patterns" | `analyze_high_risk_customer_patterns` | deep_analysis |
| `customer_prediction` | "Predict churn for 7590-VHVEG" | `predict_churn_for_customer` | prediction |
| `context_followup` | "Tell me about retained" → "yes" | (context maintained) | context_awareness |
| `financial_impact` | "Financial impact of high-risk?" | `get_high_risk_customers_with_financial_impact` | financial_analysis |

## Output Example

```
🚀 STARTING AGENT EVALUATION SUITE
════════════════════════════════════════════════════════════

📋 Running 8 test cases...

════════════════════════════════════════════════════════════
🧪 TEST: retained_vs_high_risk_comparison
📝 Query: Compare top 5 retained vs top 5 high risk customers
════════════════════════════════════════════════════════════

🔧 Tools Used: compare_retained_vs_high_risk_customers
🔄 Iterations: 1
✅ Expected tool used: compare_retained_vs_high_risk_customers

👨‍⚖️  Running LLM Judge Evaluation...
📊 Judge Score: 5/5
   Accuracy: ✅
   Relevance: ✅
   Completeness: ✅
   Professional: ✅
   Reason: Perfect response with all metrics and clear comparison

✅ TEST PASSED
════════════════════════════════════════════════════════════

📊 TEST REPORT SUMMARY
════════════════════════════════════════════════════════════

✅ Passed: 7/8 (87.5%)
⚠️  Warnings: 1/8
❌ Failed: 0/8 (0.0%)

📈 Average Judge Score: 4.63/5
🔄 Average Iterations: 2.1

📑 By Category:
   basic_query: 1/1
   filtered_query: 1/1
   prediction: 2/2
   comparison: 1/1
   deep_analysis: 1/1
   context_awareness: 0/1
   financial_analysis: 1/1
```

## Adding New Tests

Add to `TEST_CASES` list in `test_agent_evaluation.py`:

```python
{
    "id": "your_test_id",
    "query": "Your test query",
    "expected_tools": ["tool_name"],
    "expected_content": ["keyword1", "keyword2"],
    "category": "your_category",
    "max_iterations": 5  # Optional
}
```

## Integration with CI/CD

Add to your `.github/workflows/test.yml`:

```yaml
- name: Run Agent Evaluation Tests
  run: |
    pytest tests/test_agent_evaluation.py -v
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Metrics Tracked

- **Pass Rate**: Percentage of tests passing
- **Average Judge Score**: Mean LLM evaluation score (1-5)
- **Average Iterations**: Mean tool calls per query
- **Tool Selection Accuracy**: % using expected tools
- **Content Completeness**: % containing expected keywords
- **Iteration Efficiency**: % completing under iteration limit

## Thresholds

- Basic queries: ≥80% pass rate
- Predictions: ≥70% pass rate
- Comparisons: ≥80% pass rate, ≤5 iterations
- Deep analysis: ≥70% pass rate
- Financial analysis: ≥80% pass rate

## Troubleshooting

**Issue: Tests fail with "Agent crashed"**
- Check `OPENAI_API_KEY` is set
- Verify database exists at `data/churn_data.db`
- Check model file exists at `models/churn_model.pkl`

**Issue: High iteration counts**
- Review system prompt for tool selection guidance
- Check if specialized tools are being preferred
- Consider increasing `AGENT_MAX_ITERATIONS` in config

**Issue: Low judge scores**
- Review agent responses for hallucinations
- Check tool outputs contain expected data
- Verify response formatting is clean

## Future Enhancements

- [ ] Add regression tests (compare against baseline)
- [ ] Track evaluation metrics over time
- [ ] Add adversarial test cases (edge cases, errors)
- [ ] Implement A/B testing for prompt variations
- [ ] Add performance benchmarks (latency, token usage)
