"""Quick test of the improved agent with deep analysis capability"""
import sys
sys.path.insert(0, 'src')

from agents.churn_agent import ChurnAgent

# Initialize agent
print("🔄 Initializing agent...\n")
agent = ChurnAgent()  # No singleton pattern in ChurnAgent

# Test query that should trigger deep analysis
print("="*80)
print("Testing: 'yes further analysis and more recommendations'")
print("="*80)

response = agent.query("yes further analysis and more recommendations on high risk customers")

print("\n📊 AGENT RESPONSE:\n")
print(response)
print("\n" + "="*80)
