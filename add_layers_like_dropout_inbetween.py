import timm
import torch.nn as nn

# Load the model
model = timm.create_model('vgg16', pretrained=True)  # Replace 'model_name' with your model

# Example: Adding dropout after the 4th layer of the features block
if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
    # Assuming 'model.features' is the sequential block where you want to add dropout
    new_features = nn.Sequential(
        *list(model.features.children())[:4],  # Layers before the dropout
        nn.Dropout(p=0.5),  # Dropout layer
        *list(model.features.children())[4:]  # Layers after the dropout
    )
    model.features = new_features
print(model)
