#! /bin/zsh

echo "Setup the environments..."

# Check if venv diretory already exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Skip creating virtual environment."
fi

echo "Install python libraries..."
source ./venv/bin/activate
pip3 install -r requirements.txt