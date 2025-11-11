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

So far we are using git cpu for training the first model. github/workflows/train-gpu.yml file in the project is added to store github CPU config and used automatically for training the model on github server around the clock. So no need to run the training locally.

In train.py, the GPU vs CPU decision happens automatically, controlled by this line : device = pick_device(cfg.get("device", "auto")) . In the case you want to train the model locally on cpu or gpu, you can change the "**auto**" to “**cpu**” or “**cuda**” and use model parameter stored in configs/recommended.yaml

.venv\Scripts\Activate.ps1

python train.py --config configs\recommended.yaml


That loads the test set, restores the model, and prints averaged metrics (distogram/torsion/FAPE and priors, plus the combined loss). It uses the exact same loss composition as training, but with no gradient updates. (our test loop mirrors the training losses: distogram/torsion/FAPE and optional Cn/membrane/pore terms.)

**
