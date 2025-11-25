"""
Churn Prediction Agent using LangChain and OpenAI.

This module creates an intelligent agent that can:
1. Understand natural language queries about customer churn
2. Decide which tools to use and when
3. Execute multiple tools in sequence if needed
4. Provide contextual, actionable insights
"""

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import OPENAI_API_KEY, OPENAI_MODEL, AGENT_TEMPERATURE, AGENT_MAX_ITERATIONS, AGENT_VERBOSE
from tools.agent_tools import CHURN_TOOLS
from prompts.system_prompts import get_system_prompt


class ChurnAgent:
    """
    Intelligent agent for customer churn analysis and prediction.
    
    Uses LangChain + OpenAI to orchestrate tools and provide insights.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        temperature: float = None,
        verbose: bool = None,
        max_iterations: int = None
    ):
        """
        Initialize the Churn Agent.
        
        Args:
            api_key: OpenAI API key (defaults to config)
            model: Model name (defaults to config)
            temperature: Creativity level 0-1 (defaults to config)
            verbose: Show reasoning steps (defaults to config)
            max_iterations: Max tool calls per query (defaults to config)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or OPENAI_MODEL
        self.temperature = temperature if temperature is not None else AGENT_TEMPERATURE
        self.verbose = verbose if verbose is not None else AGENT_VERBOSE
        self.max_iterations = max_iterations or AGENT_MAX_ITERATIONS
        
        # Validate API key
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            raise ValueError(
                "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            )
        
        # Initialize components
        self.llm = None
        self.agent_executor = None
        self.memory = None
        
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain agent with tools and prompts."""
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key
        )
        
        # Create conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Build prompt template
        system_prompt = get_system_prompt(
            verbose=self.verbose,
            include_examples=True
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=CHURN_TOOLS,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=CHURN_TOOLS,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        print(f"✅ Churn Agent initialized with {len(CHURN_TOOLS)} tools")
        print(f"   Model: {self.model}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Verbose: {self.verbose}")
    
    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user query and return response with reasoning.
        
        Args:
            user_input: Natural language question or command
            
        Returns:
            Dictionary with:
            - output: The agent's response
            - intermediate_steps: List of (tool, result) tuples
            - chat_history: Conversation history
        """
        try:
            result = self.agent_executor.invoke({"input": user_input})
            
            return {
                "output": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "success": True
            }
        
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "output": f"I encountered an error: {str(e)}. Please try rephrasing your question or check the system logs.",
                "intermediate_steps": [],
                "success": False,
                "error": str(e)
            }
    
    def chat(self, user_input: str) -> str:
        """
        Simple chat interface (returns just the text response).
        
        Args:
            user_input: User's message
            
        Returns:
            Agent's text response
        """
        result = self.query(user_input)
        return result["output"]
    
    def reset_memory(self):
        """Clear conversation history."""
        self.memory.clear()
        print("🔄 Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        messages = self.memory.chat_memory.messages
        history = []
        
        for msg in messages:
            if hasattr(msg, 'type'):
                role = 'user' if msg.type == 'human' else 'assistant'
            else:
                role = 'assistant'
            
            history.append({
                'role': role,
                'content': msg.content
            })
        
        return history
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about the agent configuration.
        
        Returns:
            Dictionary with agent metadata
        """
        return {
            'model': self.model,
            'temperature': self.temperature,
            'verbose': self.verbose,
            'max_iterations': self.max_iterations,
            'num_tools': len(CHURN_TOOLS),
            'tools': [tool.name for tool in CHURN_TOOLS],
            'has_memory': self.memory is not None,
            'conversation_length': len(self.memory.chat_memory.messages) if self.memory else 0
        }


# ==================== CONVENIENCE FUNCTIONS ====================

_global_agent = None

def get_agent(reset: bool = False) -> ChurnAgent:
    """
    Get or create the global agent instance (singleton pattern).
    
    Args:
        reset: If True, create a new agent instance
        
    Returns:
        ChurnAgent instance
    """
    global _global_agent
    
    if _global_agent is None or reset:
        _global_agent = ChurnAgent()
    
    return _global_agent


def ask(question: str) -> str:
    """
    Quick function to ask the agent a question.
    
    Args:
        question: Natural language query
        
    Returns:
        Agent's response as string
    """
    agent = get_agent()
    return agent.chat(question)


def query(question: str) -> Dict[str, Any]:
    """
    Query the agent and get detailed response with reasoning.
    
    Args:
        question: Natural language query
        
    Returns:
        Full response dictionary with intermediate steps
    """
    agent = get_agent()
    return agent.query(question)


# ==================== INTERACTIVE MODE ====================

def interactive_mode():
    """
    Start an interactive chat session with the agent.
    
    Usage:
        from src.agents.churn_agent import interactive_mode
        interactive_mode()
    """
    print("=" * 60)
    print("🤖 CHURN PREDICTION AGENT - Interactive Mode")
    print("=" * 60)
    print("\nType your questions or commands. Type 'quit' or 'exit' to end.\n")
    print("Example queries:")
    print("  - What's the churn rate?")
    print("  - Show me churn for people without landline")
    print("  - Predict churn for customer 7590-VHVEG")
    print("  - Find high-risk customers")
    print("=" * 60)
    print()
    
    agent = get_agent()
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Check for reset command
            if user_input.lower() in ['reset', 'clear', 'new']:
                agent.reset_memory()
                print("✅ Conversation reset. Starting fresh!\n")
                continue
            
            # Check for help command
            if user_input.lower() in ['help', '?']:
                print("\n📚 Available Commands:")
                print("  - Ask any question about customer churn")
                print("  - 'reset' - Clear conversation history")
                print("  - 'info' - Show agent configuration")
                print("  - 'quit' - Exit interactive mode\n")
                continue
            
            # Check for info command
            if user_input.lower() == 'info':
                info = agent.get_agent_info()
                print(f"\n🤖 Agent Info:")
                print(f"  Model: {info['model']}")
                print(f"  Tools: {info['num_tools']}")
                print(f"  Conversation length: {info['conversation_length']} messages\n")
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Process query
            print("\nAgent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


# ==================== BATCH PROCESSING ====================

def batch_query(questions: List[str], reset_between: bool = False) -> List[Dict[str, Any]]:
    """
    Process multiple questions in batch.
    
    Args:
        questions: List of questions to ask
        reset_between: If True, reset memory between questions
        
    Returns:
        List of response dictionaries
    """
    agent = get_agent()
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing: {question}")
        
        result = agent.query(question)
        results.append({
            'question': question,
            'response': result['output'],
            'success': result['success']
        })
        
        if reset_between:
            agent.reset_memory()
    
    return results


if __name__ == "__main__":
    """Run interactive mode when script is executed directly."""
    interactive_mode()
