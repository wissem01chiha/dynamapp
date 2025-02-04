#!/bin/bash
TEST_DIR="tests"
echo "Discovering and running tests in the '$TEST_DIR' directory..."
OUTPUT=$(python3 -m unittest discover "$TEST_DIR" -p "*.py" 2>&1)
EXIT_CODE=$?
echo "$OUTPUT"
exit $EXIT_CODE
