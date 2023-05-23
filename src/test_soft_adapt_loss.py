import torch
import unittest
from loss.functional.helpers import kl_loss, recon_loss
from loss import SoftAdaptVAELoss
import numpy as np

# Create an instance of SoftAdaptVAELoss
loss_fn = SoftAdaptVAELoss(n=3, variant=["Normalized", "Loss Weighted"], beta=0.1)

# Define example losses
losses = [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.2)]

# Add losses to the buffer until it's full
for loss in losses:
    loss_fn.update_losses(losses[0], losses[1])

# Calculate the weighted loss
weighted_loss = loss_fn.weighted_loss(losses[0], losses[1])

# Define the expected weighted loss based on the SoftAdapt algorithm
expected_weighted_loss = losses[0] * 0.6 + losses[1] * 0.3 + losses[2] * 0.1



print("Expected weighted loss: ", expected_weighted_loss)
print("Actual weighted loss: ", weighted_loss)


