import timm

# Example: Creating a VGG16 model
MODEL_NAME = 'vgg16'
PRETRAIN = False  # Based on your setup
NUM_CLASSES = 5  # Based on your setup
DROP_OUT = 0.9  # Based on your setup

model = timm.create_model(MODEL_NAME, pretrained=PRETRAIN, num_classes=NUM_CLASSES, drop_rate=DROP_OUT)

# Printing the model architecture
print(model)
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (pre_logits): ConvMlp(
#     (fc1): Conv2d(512, 4096, kernel_size=(7, 7), stride=(1, 1))
#     (act1): ReLU(inplace=True)
#     (drop): Dropout(p=0.9, inplace=False)
#     (fc2): Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1))
#     (act2): ReLU(inplace=True)
#   )
#   (head): ClassifierHead(
#     (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
#     (drop): Dropout(p=0.9, inplace=False)
#     (fc): Linear(in_features=4096, out_features=5, bias=True)
#     (flatten): Identity()
#   )
# )

# freeze first 10 layers in vgg16
for layer_index in range(10):
    layer = model.features[layer_index]
    # if hasattr(layer, 'parameters'):
    for param in layer.parameters():
        param.requires_grad = False
