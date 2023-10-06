import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import timm


def get_model_transfer_learning(model_name, n_classes, fine_tune=True):
    # Create the model
    model_transfer = timm.create_model(model_name, pretrained=True, num_classes=n_classes)
    
    if not fine_tune:
        # Freeze all layers
        for param in model_transfer.parameters():
            param.requires_grad = False
    
    else:
        # Unfreeze the last layer (Assuming it is named 'classifier' or 'fc', this may not always hold)
        # Note: This is just a general case, different models might have different names for the last layer
        last_layer_names = ['classifier', 'fc', 'head']
        
        for name, param in model_transfer.named_parameters():
            if not any(layer_name in name for layer_name in last_layer_names):
                param.requires_grad = False
    
    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from src.data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(model_name='efficientvit_b2.r224_in1k', n_classes=23, fine_tune=True)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
