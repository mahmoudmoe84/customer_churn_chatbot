"""
Agent Evaluation Test Suite
Tests agent behavior, tool selection, and response quality
"""
import json
import sys
import os
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Make pytest optional (only needed for pytest runner)
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    print("⚠️  pytest not installed. Running in standalone mode...")

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from agents.churn_agent import ChurnAgent
from config import OPENAI_API_KEY


# ═══════════════════════════════════════════════════════
# 1. GOLDEN TEST DATASET
# ═══════════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": "basic_churn_rate",
        "query": "What is the overall churn rate?",
        "expected_tools": ["query_overall_churn_rate"],
        "expected_content": ["26.5", "churn rate", "%"],
        "category": "basic_query"
    },
    {
        "id": "filtered_churn_rate",
        "query": "What's the churn rate for Month-to-month customers?",
        "expected_tools": ["query_churn_rate_with_filter"],
        "expected_content": ["Month-to-month", "churn rate"],
        "category": "filtered_query"
    },
    {
        "id": "high_risk_identification",
        "query": "Show me top 5 high-risk customers",
        "expected_tools": ["find_high_risk_customers"],
        "expected_content": ["risk", "customer", "probability"],
        "category": "prediction"
    },
    {
        "id": "retained_vs_high_risk_comparison",
        "query": "Compare top 5 retained vs top 5 high risk customers",
        "expected_tools": ["compare_retained_vs_high_risk_customers"],
        "expected_content": ["retained", "high-risk", "monthly", "revenue"],
        "category": "comparison",
        "max_iterations": 3  # Should complete in 1-3 iterations
    },
    {
        "id": "pattern_analysis",
        "query": "Analyze high-risk customer patterns and give me recommendations",
        "expected_tools": ["analyze_high_risk_customer_patterns"],
        "expected_content": ["contract", "payment", "recommendation"],
        "category": "deep_analysis"
    },
    {
        "id": "customer_prediction",
        "query": "Predict churn for customer 7590-VHVEG",
        "expected_tools": ["predict_churn_for_customer"],
        "expected_content": ["7590-VHVEG", "probability", "risk"],
        "category": "prediction"
    },
    {
        "id": "context_followup",
        "query": "Tell me about retained customers",
        "followup": "yes",  # Should continue talking about retained customers
        "expected_tools": ["get_top_retained_customers_financial_analysis"],
        "expected_content": ["retained", "customer"],
        "category": "context_awareness"
    },
    {
        "id": "financial_impact",
        "query": "What's the financial impact of high-risk customers?",
        "expected_tools": ["get_high_risk_customers_with_financial_impact"],
        "expected_content": ["revenue", "lifetime value", "risk"],
        "category": "financial_analysis"
    }
]


# ═══════════════════════════════════════════════════════
# 2. LLM-AS-A-JUDGE EVALUATOR
# ═══════════════════════════════════════════════════════

class LLMJudge:
    """Uses GPT-4 to evaluate agent response quality"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Cheaper and faster for evaluation
            temperature=0,
            api_key=OPENAI_API_KEY
        )
    
    def evaluate_response(self, query: str, agent_response: str, expected_content: list) -> dict:
        """
        Evaluates agent response on multiple criteria
        
        Returns:
            {
                "score": int (1-5),
                "accuracy": bool,
                "relevance": bool,
                "completeness": bool,
                "professional": bool,
                "reason": str
            }
        """
        prompt = f"""You are a strict QA auditor evaluating a customer churn analysis chatbot.

USER QUERY: {query}
AGENT RESPONSE: {agent_response}
EXPECTED CONTENT: {', '.join(expected_content)}

Evaluate on these criteria:

1. ACCURACY: Does the response contain factual data? No hallucinations?
2. RELEVANCE: Does it directly answer the user's question?
3. COMPLETENESS: Are expected elements present (numbers, insights, recommendations)?
4. PROFESSIONALISM: Is the tone appropriate and clear?

Return ONLY a JSON object with this EXACT format (no markdown, no extra text):
{{
    "score": <int 1-5>,
    "accuracy": <true/false>,
    "relevance": <true/false>,
    "completeness": <true/false>,
    "professional": <true/false>,
    "reason": "<brief explanation>"
}}

Score guide:
5 = Perfect response
4 = Good with minor issues
3 = Acceptable but missing key elements
2 = Poor, major issues
1 = Completely failed
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = json.loads(response.content)
            return result
        except Exception as e:
            return {
                "score": 0,
                "accuracy": False,
                "relevance": False,
                "completeness": False,
                "professional": False,
                "reason": f"Judge evaluation failed: {str(e)}"
            }


# ═══════════════════════════════════════════════════════
# 3. AGENT TEST RUNNER
# ═══════════════════════════════════════════════════════

class AgentTester:
    """Runs automated tests on the churn agent"""
    
    def __init__(self):
        self.agent = ChurnAgent()
        self.judge = LLMJudge()
        self.results = []
    
    def run_test_case(self, test_case: dict) -> dict:
        """Run a single test case and return results"""
        test_id = test_case["id"]
        query = test_case["query"]
        
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {test_id}")
        print(f"📝 Query: {query}")
        print(f"{'='*60}")
        
        result = {
            "test_id": test_id,
            "query": query,
            "category": test_case["category"],
            "success": False,
            "agent_response": None,
            "intermediate_steps": [],
            "tools_used": [],
            "iteration_count": 0,
            "judge_score": None,
            "errors": []
        }
        
        try:
            # Run agent and capture intermediate steps
            agent_executor = self.agent.agent_executor
            
            response = agent_executor.invoke(
                {"input": query},
                return_only_outputs=False
            )
            
            # Extract results
            result["agent_response"] = response.get("output", "")
            result["intermediate_steps"] = response.get("intermediate_steps", [])
            
            # Extract tool usage
            if result["intermediate_steps"]:
                result["tools_used"] = [
                    step[0].tool for step in result["intermediate_steps"]
                ]
                result["iteration_count"] = len(result["intermediate_steps"])
            
            # Check for errors in response
            if "Error" in result["agent_response"] or "error" in result["agent_response"].lower():
                result["errors"].append("Agent response contains error message")
                result["success"] = False
            else:
                result["success"] = True
            
            # Print intermediate steps
            print(f"\n🔧 Tools Used: {', '.join(result['tools_used']) if result['tools_used'] else 'None'}")
            print(f"🔄 Iterations: {result['iteration_count']}")
            
            # Validate tool selection
            expected_tools = test_case.get("expected_tools", [])
            if expected_tools:
                tools_matched = any(
                    expected_tool in result["tools_used"] 
                    for expected_tool in expected_tools
                )
                if tools_matched:
                    print(f"✅ Expected tool used: {', '.join(expected_tools)}")
                else:
                    print(f"⚠️  Expected tool NOT used. Expected: {expected_tools}, Got: {result['tools_used']}")
                    result["errors"].append(f"Wrong tool selected")
            
            # Check iteration limit
            max_iterations = test_case.get("max_iterations", 15)
            if result["iteration_count"] >= max_iterations:
                print(f"⚠️  WARNING: Hit max iterations ({max_iterations})")
                result["errors"].append("Hit iteration limit")
            
            # Content validation
            expected_content = test_case.get("expected_content", [])
            missing_content = [
                content for content in expected_content
                if content.lower() not in result["agent_response"].lower()
            ]
            if missing_content:
                print(f"⚠️  Missing expected content: {', '.join(missing_content)}")
                result["errors"].append(f"Missing content: {missing_content}")
            
            # LLM Judge Evaluation
            print("\n👨‍⚖️  Running LLM Judge Evaluation...")
            judge_result = self.judge.evaluate_response(
                query=query,
                agent_response=result["agent_response"],
                expected_content=expected_content
            )
            result["judge_score"] = judge_result
            
            print(f"📊 Judge Score: {judge_result['score']}/5")
            print(f"   Accuracy: {'✅' if judge_result['accuracy'] else '❌'}")
            print(f"   Relevance: {'✅' if judge_result['relevance'] else '❌'}")
            print(f"   Completeness: {'✅' if judge_result['completeness'] else '❌'}")
            print(f"   Professional: {'✅' if judge_result['professional'] else '❌'}")
            print(f"   Reason: {judge_result['reason']}")
            
            # Final verdict
            if result["success"] and judge_result["score"] >= 4 and not result["errors"]:
                print(f"\n✅ TEST PASSED")
            elif judge_result["score"] >= 3:
                print(f"\n⚠️  TEST PASSED WITH WARNINGS")
            else:
                print(f"\n❌ TEST FAILED")
                result["success"] = False
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(str(e))
            print(f"\n❌ TEST CRASHED: {str(e)}")
        
        print(f"\n{'='*60}\n")
        return result
    
    def run_all_tests(self, categories=None) -> dict:
        """Run all test cases or specific categories"""
        print("\n" + "="*60)
        print("🚀 STARTING AGENT EVALUATION SUITE")
        print("="*60)
        
        # Filter test cases by category if specified
        test_cases = TEST_CASES
        if categories:
            test_cases = [tc for tc in TEST_CASES if tc["category"] in categories]
        
        print(f"\n📋 Running {len(test_cases)} test cases...")
        
        # Run all tests
        for test_case in test_cases:
            result = self.run_test_case(test_case)
            self.results.append(result)
        
        # Generate summary report
        return self.generate_report()
    
    def generate_report(self) -> dict:
        """Generate comprehensive test report"""
        from datetime import datetime
        
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r["success"] and not r["errors"])
        failed = sum(1 for r in self.results if not r["success"])
        warnings = total_tests - passed - failed
        
        avg_score = sum(
            r["judge_score"]["score"] for r in self.results if r["judge_score"]
        ) / total_tests if total_tests > 0 else 0
        
        avg_iterations = sum(
            r["iteration_count"] for r in self.results
        ) / total_tests if total_tests > 0 else 0
        
        # Build report string
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📊 TEST REPORT SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\n✅ Passed: {passed}/{total_tests} ({passed/total_tests*100:.1f}%)")
        report_lines.append(f"⚠️  Warnings: {warnings}/{total_tests}")
        report_lines.append(f"❌ Failed: {failed}/{total_tests} ({failed/total_tests*100:.1f}%)")
        report_lines.append(f"\n📈 Average Judge Score: {avg_score:.2f}/5")
        report_lines.append(f"🔄 Average Iterations: {avg_iterations:.1f}")
        
        # Category breakdown
        report_lines.append("\n📑 By Category:")
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["success"] and not result["errors"]:
                categories[cat]["passed"] += 1
        
        for cat, stats in categories.items():
            report_lines.append(f"   {cat}: {stats['passed']}/{stats['total']}")
        
        # Failed tests detail
        if failed > 0:
            report_lines.append("\n❌ Failed Tests:")
            for result in self.results:
                if not result["success"]:
                    report_lines.append(f"   - {result['test_id']}: {', '.join(result['errors'])}")
        
        # Detailed test results
        report_lines.append("\n" + "=" * 60)
        report_lines.append("📋 DETAILED TEST RESULTS")
        report_lines.append("=" * 60)
        for result in self.results:
            status = "✅ PASS" if result["success"] and not result["errors"] else "⚠️ WARN" if result["errors"] and result["success"] else "❌ FAIL"
            report_lines.append(f"\n{status} | {result['test_id']}")
            report_lines.append(f"   Category: {result['category']}")
            report_lines.append(f"   Query: {result['query']}")
            report_lines.append(f"   Tools Used: {', '.join(result['tools_used']) if result['tools_used'] else 'None'}")
            report_lines.append(f"   Iterations: {result['iteration_count']}")
            if result['judge_score']:
                report_lines.append(f"   Judge Score: {result['judge_score']['score']}/5")
                report_lines.append(f"   Judge Reason: {result['judge_score']['reason']}")
            if result['errors']:
                report_lines.append(f"   Errors: {', '.join(result['errors'])}")
        
        report_lines.append("\n" + "=" * 60 + "\n")
        
        # Print to console
        report_text = "\n".join(report_lines)
        print("\n" + report_text)
        
        # Save to log file
        log_dir = Path(__file__).parent / "test_logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "test_results.txt"
        
        # Append to log file with separator
        with open(log_file, "a", encoding="utf-8") as f:
            if log_file.stat().st_size > 0:
                f.write("\n" + "=" * 80 + "\n")
            f.write(report_text)
            f.write("\n")
        
        print(f"💾 Test results saved to: {log_file}")
        
        return {
            "total": total_tests,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "pass_rate": passed / total_tests if total_tests > 0 else 0,
            "avg_score": avg_score,
            "avg_iterations": avg_iterations,
            "categories": categories,
            "results": self.results
        }


# ═══════════════════════════════════════════════════════
# 4. PYTEST TEST FUNCTIONS (only if pytest is available)
# ═══════════════════════════════════════════════════════

if PYTEST_AVAILABLE:
    @pytest.fixture(scope="module")
    def agent_tester():
        """Fixture to create agent tester instance"""
        return AgentTester()


    def test_basic_queries(agent_tester):
        """Test basic churn rate queries"""
        report = agent_tester.run_all_tests(categories=["basic_query", "filtered_query"])
        assert report["pass_rate"] >= 0.8, f"Basic queries pass rate too low: {report['pass_rate']}"


    def test_predictions(agent_tester):
        """Test prediction capabilities"""
        report = agent_tester.run_all_tests(categories=["prediction"])
        assert report["pass_rate"] >= 0.7, f"Prediction pass rate too low: {report['pass_rate']}"


    def test_comparisons(agent_tester):
        """Test comparison queries (retained vs high-risk)"""
        report = agent_tester.run_all_tests(categories=["comparison"])
        
        # Check iteration efficiency
        comparison_results = [r for r in agent_tester.results if r["category"] == "comparison"]
        max_iterations_used = max(r["iteration_count"] for r in comparison_results)
        
        assert max_iterations_used <= 5, f"Comparison took too many iterations: {max_iterations_used}"
        assert report["pass_rate"] >= 0.8, f"Comparison pass rate too low: {report['pass_rate']}"


    def test_deep_analysis(agent_tester):
        """Test pattern analysis and recommendations"""
        report = agent_tester.run_all_tests(categories=["deep_analysis"])
        assert report["pass_rate"] >= 0.7, f"Deep analysis pass rate too low: {report['pass_rate']}"


    def test_context_awareness(agent_tester):
        """Test context retention across conversation"""
        report = agent_tester.run_all_tests(categories=["context_awareness"])
        assert report["pass_rate"] >= 0.7, f"Context awareness pass rate too low: {report['pass_rate']}"


    def test_financial_analysis(agent_tester):
        """Test financial impact calculations"""
        report = agent_tester.run_all_tests(categories=["financial_analysis"])
        
        # Check for proper formatting (no concatenation issues)
        financial_results = [r for r in agent_tester.results if r["category"] == "financial_analysis"]
        for result in financial_results:
            assert "$" in result["agent_response"], "Missing currency symbol"
            # Check no concatenation like "476.80inlifetime"
            assert "inlifetime" not in result["agent_response"].lower()
        
        assert report["pass_rate"] >= 0.8, f"Financial analysis pass rate too low: {report['pass_rate']}"


# ═══════════════════════════════════════════════════════
# 5. COMMAND-LINE RUNNER
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run agent evaluation tests")
    parser.add_argument(
        "--category",
        type=str,
        help="Test specific category only",
        choices=["basic_query", "filtered_query", "prediction", "comparison", 
                 "deep_analysis", "context_awareness", "financial_analysis"]
    )
    parser.add_argument(
        "--test-id",
        type=str,
        help="Run specific test by ID"
    )
    
    args = parser.parse_args()
    
    tester = AgentTester()
    
    if args.test_id:
        # Run single test
        test_case = next((tc for tc in TEST_CASES if tc["id"] == args.test_id), None)
        if test_case:
            result = tester.run_test_case(test_case)
            tester.generate_report()
        else:
            print(f"Test ID '{args.test_id}' not found")
    else:
        # Run all or filtered tests
        categories = [args.category] if args.category else None
        report = tester.run_all_tests(categories=categories)
        
        # Exit with error code if tests failed
        if report["failed"] > 0:
            sys.exit(1)
