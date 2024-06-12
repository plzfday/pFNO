import os

import torch
from timeit import default_timer
import numpy as np
from tqdm import tqdm

from utilities import LpLoss, count_params
from model import FNO2d
from data import (
    train_loader,
    test_loader,
    y_normalizer,
)
from config import (
    RESULT_PATH,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    NTEST,
    NTRAIN,
    MODES,
    WIDTH,
    S,
    ITERATIONS,
)

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, test_loader, y_normalizer, optimizer, scheduler):
    os.makedirs(RESULT_PATH, exist_ok=True)

    train_losses = []
    test_losses = []

    criterion = LpLoss(size_average=False)

    best = 1e8

    for ep in tqdm(range(EPOCHS)):
        t1 = default_timer()

        # train
        model.train()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x).reshape(BATCH_SIZE, S, S)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = criterion(out.view(BATCH_SIZE, -1), y.view(BATCH_SIZE, -1))
            loss.backward()

            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()

        # validation
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x).reshape(BATCH_SIZE, S, S)
                out = y_normalizer.decode(out)

                test_l2 += criterion(
                    out.view(BATCH_SIZE, -1), y.view(BATCH_SIZE, -1)
                ).item()

        # save the best model
        if test_l2 < best:
            best = test_l2
            torch.save(model.state_dict(), RESULT_PATH + "/model_darcy_best.pth")

        # done with this epoch
        train_l2 /= NTRAIN
        test_l2 /= NTEST

        train_losses.append(train_l2)
        test_losses.append(test_l2)

        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2)

    np.savez(RESULT_PATH + "/darcy_train_losses.npz", train_losses, test_losses)


################################################################
# training and evaluation
################################################################
model = FNO2d(MODES, MODES, WIDTH).to(device)
print(f"Number of params: {count_params(model)}")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ITERATIONS)
y_normalizer.to(device)

train_model(model, train_loader, test_loader, y_normalizer, optimizer, scheduler)
