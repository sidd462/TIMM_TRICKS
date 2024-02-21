import timm
import torch
import torch.nn as nn

# Load the InceptionV3 model
model = timm.create_model('inception_v3', pretrained=False)
for name, module in model.named_children():
    print(name)
    

# Function to recursively add dropout layers
def add_dropout_recursive(module, dropout_rate=0.5):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):  # Targeting the place after ReLU activations
            # Need to be careful; directly adding modules here changes the model structure
            # A safer approach is to encapsulate functionality or modify the source directly
            continue
        elif len(list(child.children())) > 0:
            # Recursively apply to children modules if they have children of their own
            add_dropout_recursive(child, dropout_rate)
    # After iterating through children, add dropout if the module is a suitable container
    if hasattr(module, 'add_module'):
        module.add_module(name="Dropout", module=nn.Dropout(p=dropout_rate))

# Example of applying it to a specific block (e.g., Mixed_7c)
if hasattr(model, 'Mixed_5b'):
    add_dropout_recursive(model.Mixed_5b, 0.5)
if hasattr(model, 'Mixed_5d'):
    add_dropout_recursive(model.Mixed_5d, 0.5)
if hasattr(model, 'Mixed_7c'):
    add_dropout_recursive(model.Mixed_7c, 0.5)
    

print(model)
