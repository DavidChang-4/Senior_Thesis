import torch

# Example: a list of tensors
result = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

# Stack the tensors into one tensor
stacked = torch.stack(result)  # shape becomes (n, ...)

# Compute the mean of all elements
mean_result = torch.mean(stacked)

print(mean_result)  # Output: tensor(2.5)