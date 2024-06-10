import yaml

configs = None
with open("config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.safe_load(f)

NTRAIN = configs["n_train"]
NTEST = configs["n_test"]

BATCH_SIZE = configs["batch_size"]
LEARNING_RATE = configs["lr"]
EPOCHS = configs["epochs"]
ITERATIONS = EPOCHS * (NTRAIN // BATCH_SIZE)

MODES = configs["modes"]
WIDTH = configs["width"]

R = configs["r"]
S = int(((421 - 1) / R) + 1)

TRAIN_PATH = configs["train_path"]
TEST_PATH = configs["test_path"]

RESULT_PATH = configs["result_path"]
