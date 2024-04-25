import torch

# Example 2D tensor (rows are records, columns are features)
records_tensor = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])

# Example list of 1D arrays
arrays_list = [torch.tensor([0.5, 1.0, 1.5]),
               torch.tensor([2.0, 1.0, 0.5]),
               torch.tensor([2.0, 1.0, 0.5]),
               torch.tensor([2.0, 1.0, 0.5]),
               torch.tensor([2.0, 1.0, 0.5])]

# Convert the list of 1D arrays into a 2D tensor
combined_array = torch.stack(arrays_list).t()

# Perform broadcasting and subtraction in one step
new_record = records_tensor.unsqueeze(2)
new_arrays = combined_array.unsqueeze(0)

print(new_record.shape)
print(new_arrays.shape)

print((new_record - new_arrays)[0].T)
