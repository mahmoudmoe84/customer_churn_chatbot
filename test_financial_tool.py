"""Test the new financial impact tool"""
import sys
sys.path.insert(0, 'src')

from agents.churn_agent import ChurnAgent

# Initialize agent
print("🔄 Initializing agent...\n")
agent = ChurnAgent()

# Test the financial query
print("="*80)
print("Testing: 'get me top 5 customers at risk with their charges'")
print("="*80)

response = agent.query("get me top 5 customers at risk and their total charge and how much they represent out of total population charge")

print("\n📊 AGENT RESPONSE:\n")
print(response.get('output', response))
print("\n" + "="*80)
