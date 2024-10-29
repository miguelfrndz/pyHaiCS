#!/bin/bash

# Bash Script for running all pyHaiCS tests simultaneoulsy

source pyHaiCS/config/bash_colors.sh # Contains terminal color options

echo -e "\n================================================"
echo -e "Testing ${BIBlue}pyHaiCS${Color_Off} Modules..."
echo -e "================================================\n"

cd pyHaiCS/tests # CD into test directory

echo -e "${BIBlue}Running Import & Namespace Tests...${Color_Off}"
python -m unittest -v test_imports.py

echo -e "${BIBlue}Running Integrator Tests...${Color_Off}"
python -m unittest -v test_integrators.py

echo -e "${BIBlue}\nFinished Running All Tests!${Color_Off}\n"
exit 0