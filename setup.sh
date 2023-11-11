echo "Go to project dir: "$(dirname "$(realpath $0)")
cd $(dirname "$(realpath $0)")

echo "Creating python environment"
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies"
pip install -r requirements.txt

echo "Add this project to env as editable package"
pip install -e .

echo "Done"
