#!/bin/bash

# Bash Script for running all pyHaiCS tests simultaneoulsy

source pyHaiCS/config/bash_colors.sh # Contains terminal color options

# Remove old cached Python files only if --no-cache flag is set
if [[ "$@" == *"--no-cache"* ]]; then
    echo -e "${BIGreen}\nCleaning Up Old Python Cached Files...${Color_Off}"
    pyclean () {
        find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
    }
    cd ./
    pyclean
fi

echo -e "\n================================================"
echo -e "Testing ${BIPurple}pyHaiCS${Color_Off} Modules..."
echo -e "================================================"

cd pyHaiCS/tests # CD into test directory

echo -e "\n${BIBlue}Running Import & Namespace Tests...${Color_Off}"
uv run -m unittest -v test_imports.py

echo -e "\n${BIBlue}Running Integrator Tests...${Color_Off}"
uv run -m unittest -v test_integrators.py

echo -e "\n${BIBlue}Running Sampler Tests...${Color_Off}"
uv run -m unittest -v test_samplers.py

echo -e "\n${BIGreen}\nFinished Running All Tests!${Color_Off}\n"
exit 0