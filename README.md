# First time dependencies setup

All dependecies are listed in requirments.txt. First time you running his project, you need to make a local venv, activate it and install dependencies locally on you loptop, so in terminal run this code:

git pull
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

In case, you add new library to dependencies, just update the requirments.txt by running this code:

pip freeze > requirements.txt

git add requirements.txt
git commit -m "Update dependencies"
git pushviroporinaf_mini

py eval.py
