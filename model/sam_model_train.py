from torch.utils.data import Dataset
from transformers import SamProcessor
from torch.utils.data import DataLoader
from transformers import SamModel 
from torch.optim import Adam
import monai

from sklearn.metrics import jaccard_score
import numpy as np

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize


class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox


def sam_model_train(train_data, val_data):
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    train_dataset = SAMDataset(dataset=train_data, processor=processor)
    val_dataset = SAMDataset(dataset=val_data, processor=processor)


    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)


    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    num_epochs = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
    

    model.eval()

    # Initialize an empty array to store IoU for each image
    ious = []

    #Pixel ACC array
    accuracies = []

    for batch in tqdm(train_dataloader):
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=1)  # Assuming logits is the output

        true_masks = batch['labels'].cpu().numpy()

       # Flatten the arrays for compatibility with jaccard_score
        predictions = predictions.flatten()
        true_masks = true_masks.flatten()

        # Calculate IoU and append to list
        iou = jaccard_score(true_masks, predictions, average='macro')  # 'macro' averages over classes
        ious.append(iou)

        correct = (predictions == true_masks).sum()
        total = len(true_masks)
        accuracy = correct / total
        accuracies.append(accuracy)

    

    mean_iou = np.mean(ious)

    mean_accuracy = np.mean(accuracies)

    return mean_iou, mean_accuracy




 


