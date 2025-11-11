# First time dependencies setup

All dependecies are listed in requirments.txt. First time you running his project, you need to make a local venv, activate it and install dependencies locally on you loptop, so in terminal run this code:

**git pull
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt**

In case, you add new library to dependencies, just update the requirments.txt by running this code:

**pip freeze > requirements.txt**

**git add requirements.txt
git commit -m "Update dependencies"
git pushviroporinaf_mini**

The training pipeline works like this:

1. train.py reads train.jsonl → gets a list of all sequences and their paths.
2. For each entry, it loads the corresponding .npz feature file.
3. Those features are batched and sent into the model for training.

To train the model, you need to run following code, using model parameter stored in configs/recommended.yaml

You **should absolutely run your code inside your virtual environment** (the “env”) if you want GPU training to work reliably:

**.\venv\Scripts\activate**

python train.py --config configs\recommended.yaml


After training, test the result by running following:

**python test.py --config configs\recommended.yaml**

That loads the test set, restores the model, and prints averaged metrics (distogram/torsion/FAPE and priors, plus the combined loss). It uses the exact same loss composition as training, but with no gradient updates. (our test loop mirrors the training losses: distogram/torsion/FAPE and optional Cn/membrane/pore terms.)

**
