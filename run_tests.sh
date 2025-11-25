#!/bin/bash
# Quick test runner for agent evaluation

echo "🚀 Agent Evaluation Test Runner"
echo "================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "✅ Activating virtual environment..."
    source .venv/bin/activate
    PYTHON_CMD="python"
elif [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
    PYTHON_CMD="python"
else
    echo "⚠️  No virtual environment found. Using system Python3..."
    PYTHON_CMD="python3"
fi

echo ""

# Function to show menu
show_menu() {
    echo "Select test mode:"
    echo "1) Run all tests"
    echo "2) Run basic queries only"
    echo "3) Run comparison tests"
    echo "4) Run prediction tests"
    echo "5) Run deep analysis tests"
    echo "6) Run financial analysis tests"
    echo "7) Run specific test by ID"
    echo "8) Run with pytest (full)"
    echo "0) Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [0-8]: " choice
    
    case $choice in
        1)
            echo "Running all tests..."
            $PYTHON_CMD tests/test_agent_evaluation.py
            ;;
        2)
            echo "Running basic query tests..."
            $PYTHON_CMD tests/test_agent_evaluation.py --category basic_query
            ;;
        3)
            echo "Running comparison tests..."
            $PYTHON_CMD tests/test_agent_evaluation.py --category comparison
            ;;
        4)
            echo "Running prediction tests..."
            $PYTHON_CMD tests/test_agent_evaluation.py --category prediction
            ;;
        5)
            echo "Running deep analysis tests..."
            $PYTHON_CMD tests/test_agent_evaluation.py --category deep_analysis
            ;;
        6)
            echo "Running financial analysis tests..."
            $PYTHON_CMD tests/test_agent_evaluation.py --category financial_analysis
            ;;
        7)
            read -p "Enter test ID: " test_id
            echo "Running test: $test_id"
            $PYTHON_CMD tests/test_agent_evaluation.py --test-id "$test_id"
            ;;
        8)
            echo "Running pytest..."
            pytest tests/test_agent_evaluation.py -v
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
