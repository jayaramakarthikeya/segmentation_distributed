import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence
from torchvision.transforms import functional as F
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM
from dataclasses import dataclass
import dataclasses

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

working_dir = "."

def save_model_checkpoint(model, cp_name):
    torch.save(model.state_dict(), os.path.join(working_dir, cp_name))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Load model from saved checkpoint
def load_model_from_checkpoint(model, ckp_path):
    return model.load_state_dict(
        torch.load(
            ckp_path,
            map_location=get_device(),
        )
    )

# Send the Tensor or Model (input argument x) to the right device
# for this notebook. i.e. if GPU is enabled, then send to GPU/CUDA
# otherwise send to CPU.
def to_device(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x.cpu()

# Simple torchvision compatible transform to send an input tensor
# to a pre-specified device.
class ToDevice(torch.nn.Module):
    """
    Sends the input object to the device specified in the
    object's constructor by calling .to(device) on the object.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, img):
        return img.to(self.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
    
def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")
# end if

def close_figures():
    while len(plt.get_fignums()) > 0:
        plt.close()
    # end while
# end def

def print_title(title):
    title_len = len(title)
    dashes = ''.join(["-"] * title_len)
    print(f"\n{title}\n{dashes}")
# end def

# ImageToPatches returns multiple flattened square patches from an
# input image tensor.
class ImageToPatches(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    # end def
    
    def forward(self, x):
        assert len(x.size()) == 4
        y = self.unfold(x)
        y = y.permute(0, 2, 1)
        print(f"Shape after ImageToPatches: {y.shape}")  # Debugging line
        return y
    # end def
# end class

# print_title("ImageToPatches")
# i2p = ImageToPatches(8, 4)
# x = torch.arange(64).reshape(8, 8).float().reshape(1, 1, 8, 8)
# y = i2p(x)
# print(x)
# print(y)
# print(f"{x.shape} -> {y.shape}")

# print_title("nn.Fold")
# fold = nn.Fold(output_size=(8, 8), kernel_size=4, stride=4)
# y = y.permute(0, 2, 1)
# z = fold(y)
# print(z)
# print(f"{y.shape} -> {z.shape}")

# x = torch.randn(10, 3, 224, 224)
# i2p = ImageToPatches(224, 16)
# y = i2p(x)
# print(x.shape, y.shape)


# The PatchEmbedding layer takes multiple image patches in (B,T,Cin) format
# and returns the embedded patches in (B,T,Cout) format.
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        # A single Layer is used to map all input patches to the output embedding dimension.
        # i.e. each image patch will share the weights of this embedding layer.
        self.embed_layer = nn.Linear(in_features=in_channels, out_features=embed_size)
    # end def
    
    def forward(self, x):
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x
    # end def
# end class

# print_title("PatchEmbedding")
# x = torch.randn(10, 196, 768)
# pe = PatchEmbedding(768, 256)
# y = pe(x)
# print(f"{x.shape} -> {y.shape}")


class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        """in_channels is the number of input channels in the input that will be
        fed into this layer. For RGB images, this value would be 3.
        """
        super().__init__()
        self.i2p = ImageToPatches(image_size, patch_size)
        self.pe = PatchEmbedding(patch_size * patch_size * in_channels, embed_size)
        num_patches = (image_size // patch_size) ** 2
        print(f"Calculated number of patches: {num_patches}")   
        # position_embed below is the learned embedding for the position of each patch
        # in the input image. They correspond to the cosine similarity of embeddings
        # visualized in the paper "An Image is Worth 16x16 Words"
        # https://arxiv.org/pdf/2010.11929.pdf (Figure 7, Center).
        self.position_embed = nn.Parameter(torch.randn(num_patches, embed_size))
    # end def
    
    def forward(self, x):
        x = self.i2p(x)
        # print(x.shape)
        x = self.pe(x)

        print(f"Shape of x after PatchEmbedding: {x.shape}")
        print(f"Shape of position embedding: {self.position_embed.shape}")

        x = x + self.position_embed
        return x
    # end def
# end class

# print_title("VisionTransformerInput")
# x = torch.randn(10, 3, 224, 224)
# vti = VisionTransformerInput(224, 16, 3, 256)
# y = vti(x)
# print(f"{x.shape} -> {y.shape}")


# The MultiLayerPerceptron is a unit of computation. It expands the input
# to 4x the number of channels, and then contracts it back into the number
# of input channels. There's a GeLU activation in between, and the layer
# is followed by a droput layer.
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout),
        )
    # end def
    
    def forward(self, x):
        return self.layers(x)
    # end def
# end class

# print_title("MultiLayerPerceptron")
# x = torch.randn(10, 50, 60)
# mlp = MultiLayerPerceptron(60, dropout=0.2)
# y = mlp(x)
# print(f"{x.shape} -> {y.shape}")


# This is a single self-attention encoder block, which has a multi-head attention
# block within it. The MultiHeadAttention block performs communication, while the
# MultiLayerPerceptron performs computation.
class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        # self.kqv = nn.Linear(embed_size, embed_size * 3)
        self.mha = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)
    # end def
    
    def forward(self, x):
        y = self.ln1(x)
        # y = self.kqv(x)
        # (q, k, v) = torch.split(y, self.embed_size, dim=2)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x
    # end def
# end class

# print_title("SelfAttentionEncoderBlock")
# x = torch.randn(10, 20, 256)
# attention_block = SelfAttentionEncoderBlock(256, 8, dropout=0.2)
# y = attention_block(x)
# print(f"{x.shape} -> {y.shape}")


# Similar to the PatchEmbedding class, we need to un-embed the representation
# of each patch that has been produced by our transformer network. We project
# each patch (that has embed_size) dimensions into patch_size*patch_size*output_dims
# channels, and then fold all the pathces back to make it look like an image.
class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, output_dims):
        super().__init__()
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.projection = nn.Linear(embed_size, patch_size * patch_size * output_dims)
        self.fold = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)
    # end def
    
    def forward(self, x):
        B, T, C = x.shape
        x = self.projection(x)
        # x will now have shape (B, T, PatchSize**2 * OutputDims). This can be folded into
        # the desired output shape.

        # To fold the patches back into an image-like form, we need to first
        # swap the T and C dimensions to make it a (B, C, T) tensor.
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x
    # end def
# end class

# print_title("OutputProjection")
# x = torch.randn(2, 196, 256)
# op = OutputProjection(224, 16, 256, 3)
# y = op(x)
# print(f"{x.shape} -> {y.shape}")

# Define a custom IoU Metric for validating the model.
def IoUMetric(pred, gt, softmax=False):
    # Run softmax if input is logits.
    if softmax is True:
        pred = nn.Softmax(dim=1)(pred)
    # end if
    
    # Add the one-hot encoded masks for all 3 output channels
    # (for all the classes) to a tensor named 'gt' (ground truth).
    gt = torch.cat([ (gt == i) for i in range(3) ], dim=1)
    # print(f"[2] Pred shape: {pred.shape}, gt shape: {gt.shape}")

    intersection = gt * pred
    union = gt + pred - intersection

    # Compute the sum over all the dimensions except for the batch dimension.
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)
    
    # Compute the mean over the batch dimension.
    return iou.mean()

class IoULoss(nn.Module):
    def __init__(self, softmax=False):
        super().__init__()
        self.softmax = softmax
    
    # pred => Predictions (logits, B, 3, H, W)
    # gt => Ground Truth Labales (B, 1, H, W)
    def forward(self, pred, gt):
        # return 1.0 - IoUMetric(pred, gt, self.softmax)
        # Compute the negative log loss for stable training.
        return -(IoUMetric(pred, gt, self.softmax).log())
    # end def
# end class

def test_custom_iou_loss():
    #               B, C, H, W
    x = torch.rand((2, 3, 2, 2), requires_grad=True)
    y = torch.randint(0, 3, (2, 1, 2, 2), dtype=torch.long)
    z = IoULoss(softmax=True)(x, y)
    return z
# end def

class VisionTransformerForSegmentation(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels, embed_size, num_blocks, num_heads, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        
        heads = [ SelfAttentionEncoderBlock(embed_size, num_heads, dropout) for i in range(num_blocks) ]
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            VisionTransformerInput(image_size, patch_size, in_channels, embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, patch_size, embed_size, out_channels),
        )
    # end def
    
    def forward(self, x):
        x = self.layers(x)
        return x
    # end def
# end class

@dataclass
class VisionTransformerArgs:
    """Arguments to the VisionTransformerForSegmentation."""
    image_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    out_channels: int = 3
    embed_size: int = 768
    num_blocks: int = 12
    num_heads: int = 8
    dropout: float = 0.2
# end class

# Train the model for a single epoch
def train_model(model, loader, optimizer):
    to_device(model.train())
    cel = True
    if cel:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = IoULoss(softmax=True)
    # end if

    running_loss = 0.0
    running_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        optimizer.zero_grad()
        inputs = to_device(inputs)
        targets = to_device(targets)
        outputs = model(inputs)
        
        # The ground truth labels have a channel dimension (NCHW).
        # We need to remove it before passing it into
        # CrossEntropyLoss so that it has shape (NHW) and each element
        # is a value representing the class of the pixel.
        if cel:
            targets = targets.squeeze(dim=1)
        # end if
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
        running_samples += targets.size(0)
        running_loss += loss.item()
    # end for

    print("Trained {} samples, Loss: {:.4f}".format(
        running_samples,
        running_loss / (batch_idx+1),
    ))
# end def

# Define training loop. This will train the model for multiple epochs.
#
# epochs: A tuple containing the start epoch (inclusive) and end epoch (exclusive).
#         The model is trained for [epoch[0] .. epoch[1]) epochs.
#
def train_loop(model, loader, epochs, optimizer, scheduler):
    epoch_i, epoch_j = epochs
    print("####TRAINING#####")
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(model, loader, optimizer)
        # with torch.inference_mode():
        #     # Display the plt in the final training epoch.
        #     # (epoch == epoch_j-1)
        #     print_test_dataset_masks(model, test_inputs, test_targets, epoch=epoch, save_path=save_path, show_plot=True)
        # # end with

        if scheduler is not None:
            scheduler.step()
        # end if
        print("")
    # end for
# end def

def prediction_accuracy(ground_truth_labels, predicted_labels):
    eq = ground_truth_labels == predicted_labels
    return eq.sum().item() / predicted_labels.numel()
    
def print_test_dataset_masks(model, test_pets_targets, test_pets_labels, epoch, save_path, show_plot):
    to_device(model.eval())
    predictions = model(to_device(test_pets_targets))
    test_pets_labels = to_device(test_pets_labels)
    # print("Predictions Shape: {}".format(predictions.shape))
    pred = nn.Softmax(dim=1)(predictions)

    pred_labels = pred.argmax(dim=1)
    # Add a value 1 dimension at dim=1
    pred_labels = pred_labels.unsqueeze(1)
    # print("pred_labels.shape: {}".format(pred_labels.shape))
    pred_mask = pred_labels.to(torch.float)
    
    # accuracy = prediction_accuracy(test_pets_labels, pred_labels)
    iou = to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND))
    iou_accuracy = iou(pred_mask, test_pets_labels)
    pixel_metric = to_device(TM.classification.MulticlassAccuracy(3, average='micro'))
    pixel_accuracy = pixel_metric(pred_labels, test_pets_labels)
    custom_iou = IoUMetric(pred, test_pets_labels)
    title = f'Epoch: {epoch:02d}, Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Custom IoU: {custom_iou:.4f}]'
    print(title)
    # print(f"Accuracy: {accuracy:.4f}")

    # Close all previously open figures.
    close_figures()
    
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title, fontsize=12)

    fig.add_subplot(3, 1, 1)
    plt.imshow(t2img(torchvision.utils.make_grid(test_pets_targets, nrow=7)))
    plt.axis('off')
    plt.title("Targets")

    fig.add_subplot(3, 1, 2)
    plt.imshow(t2img(torchvision.utils.make_grid(test_pets_labels.float() / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    plt.imshow(t2img(torchvision.utils.make_grid(pred_mask / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Predicted Labels")
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"epoch_{epoch:02}.png"), format="png", bbox_inches="tight", pad_inches=0.4)
    # end if
    
    if show_plot is False:
        close_figures()
    else:
        plt.show()
    # end if
# end def

def test_dataset_accuracy(model, loader):
    to_device(model.eval())
    iou = to_device(TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND))
    pixel_metric = to_device(TM.classification.MulticlassAccuracy(3, average='micro'))
    
    iou_accuracies = []
    pixel_accuracies = []
    custom_iou_accuracies = []
    
    print_model_parameters(model)

    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        inputs = to_device(inputs)
        targets = to_device(targets)
        predictions = model(inputs)
        
        pred_probabilities = nn.Softmax(dim=1)(predictions)
        pred_labels = predictions.argmax(dim=1)

        # Add a value 1 dimension at dim=1
        pred_labels = pred_labels.unsqueeze(1)
        # print("pred_labels.shape: {}".format(pred_labels.shape))
        pred_mask = pred_labels.to(torch.float)

        iou_accuracy = iou(pred_mask, targets)
        # pixel_accuracy = pixel_metric(pred_mask, targets)
        pixel_accuracy = pixel_metric(pred_labels, targets)
        custom_iou = IoUMetric(pred_probabilities, targets)
        iou_accuracies.append(iou_accuracy.item())
        pixel_accuracies.append(pixel_accuracy.item())
        custom_iou_accuracies.append(custom_iou.item())
        
        del inputs
        del targets
        del predictions
    # end for
    
    iou_tensor = torch.FloatTensor(iou_accuracies)
    pixel_tensor = torch.FloatTensor(pixel_accuracies)
    custom_iou_tensor = torch.FloatTensor(custom_iou_accuracies)
    
    print("Test Dataset Accuracy")
    print(f"Pixel Accuracy: {pixel_tensor.mean():.4f}, IoU Accuracy: {iou_tensor.mean():.4f}, Custom IoU Accuracy: {custom_iou_tensor.mean():.4f}")


def vit_train(train_data, val_data):
    vit_args = dataclasses.asdict(VisionTransformerArgs())

    model = VisionTransformerForSegmentation(**vit_args)

    print_model_parameters(model)

    to_device(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.8) 

    train_loop(model, train_data, (1, 51), optimizer, scheduler)

    with torch.inference_mode():
        test_dataset_accuracy(model, val_data)



    

