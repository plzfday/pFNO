from torch.utils.data import DataLoader, TensorDataset
from utilities import MatReader, UnitGaussianNormalizer
from config import (
    TRAIN_PATH,
    TEST_PATH,
    NTRAIN,
    NTEST,
    BATCH_SIZE,
    R,
    S,
)

################################################################
# load data and data normalization
################################################################
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field("coeff")[:NTRAIN, ::R, ::R][:, :S, :S]
y_train = reader.read_field("sol")[:NTRAIN, ::R, ::R][:, :S, :S]

reader.load_file(TEST_PATH)
x_test = reader.read_field("coeff")[:NTEST, ::R, ::R][:, :S, :S]
y_test = reader.read_field("sol")[:NTEST, ::R, ::R][:, :S, :S]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(NTRAIN, S, S, 1)
x_test = x_test.reshape(NTEST, S, S, 1)

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_loader = DataLoader(
    TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False
)
