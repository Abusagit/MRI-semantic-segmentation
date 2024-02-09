import torch
from torch import nn
import os

from pathlib import Path
import torchvision
import torchmetrics as TM

from matplotlib import pyplot as plt
import pickle


from tqdm.auto import tqdm


from training.utils import close_figures, t2img, BimapClasses, save_model_checkpoint, plot_inputs_targets_predictions, create_gif_from_images
from training.metric import IoUMetric




def train_model(model, train_loader, eval_loader, test_loader, optimizer, num_epochs, criterion, device, use_cross_entropy,
                save_path: Path, scheduler=None,
                ):

    (save_path / "TRAIN").mkdir(exist_ok=True)

    (save_path / "TEST").mkdir(exist_ok=True)
    (save_path / "VAL").mkdir(exist_ok=True)
    
    
    model.train()
    
    progess_bar = tqdm(range(1, num_epochs + 1), total=num_epochs, desc="[TRAIN]")
    
    
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []
    
    
    torchmetrics_JaccardIndex = TM.classification.MulticlassJaccardIndex(2, average='micro', ignore_index=BimapClasses.BACKGROUND).to(device)
    torchmetrics_Accuracy = TM.classification.MulticlassAccuracy(2, average='micro').to(device)
    
    
    for epoch in progess_bar:
        model.train()
        running_loss = 0.0
        running_samples = 0
        
        train_ious = []

        inputs_for_plot = None
        targets_for_plot = None
        predictions_for_plot = None
        
        for batch_number, (inputs, targets) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            
            logits = model(inputs)
            
            pred = logits.softmax(1)
            # The ground truth labels have a channel dimension (NCHW).
            # We need to remove it before passing it into
            # CrossEntropyLoss so that it has shape (NHW) and each element
            # is a value representing the class of the pixel.
            train_ious.append(IoUMetric(pred=pred, gt=targets))

            if use_cross_entropy:
                targets_criterion = targets.squeeze(dim=1)
            else:
                targets_criterion = targets_criterion
            
            # print(logits.shape, pred.shape, targets.shape)
            
            loss = criterion(logits, targets_criterion)
            loss.backward()
            optimizer.step()
            
            if batch_number == 1:
                inputs_for_plot = inputs
                targets_for_plot = targets
                predictions_for_plot = pred.argmax(1, keepdims=True).float()
            
            
            running_samples += targets.size(0)
            running_loss += loss.item()
        
        # print(inputs.size(0), targets.size(0))
        
        epoch_loss = running_loss / batch_number
        progess_bar.set_description("[TRAIN] {} samples, Loss: {:.4f}".format(
                                                        running_samples,
                                                        epoch_loss
                                                        ))

        
        title, val_pixel_accuracy, val_iou_accuracy, val_custom_iou, val_total_loss = validate_model(model, eval_loader, epoch, save_path=save_path, show_plot=True, device=device,
                                                                                        is_test=False, torchmetrics_Accuracy=torchmetrics_Accuracy,
                                                                                        torchmetrics_JaccardIndex=torchmetrics_JaccardIndex,
                                                                                        criterion=criterion, use_cross_entropy=use_cross_entropy)
        
        val_scores.append(val_custom_iou)
        val_losses.append(val_total_loss)
        
        train_losses.append(epoch_loss)
        train_scores.append(torch.FloatTensor(train_ious).mean())
        
        title = f'[TRAIN] Epoch: {epoch}, Loss: {epoch_loss:.4f}, Accuracy[Custom IoU: {train_scores[-1]:.4f}]'
        plot_inputs_targets_predictions(inputs=inputs_for_plot, targets=targets_for_plot, predictions=predictions_for_plot, 
                                            save_path=str(save_path / "TRAIN" / f"epoch_{epoch}.png"),
                                            title=title, show_plot=False)

        if scheduler is not None:
            scheduler.step()
    
    title, test_pixel_accuracy, test_iou_accuracy, test_custom_iou, test_total_loss = validate_model(model, test_loader, epoch, save_path=save_path, show_plot=True, device=device,
                                                                                        is_test=True, torchmetrics_Accuracy=torchmetrics_Accuracy,
                                                                                        torchmetrics_JaccardIndex=torchmetrics_JaccardIndex,
                                                                                        criterion=criterion, use_cross_entropy=use_cross_entropy)
    
    
    save_model_checkpoint(model, "last.pth", working_dir=save_path)
    
    
    report_dict = dict(
        test_iou = test_custom_iou,
        train_losses=train_losses,
        train_scores=train_scores,
        val_scores=val_scores,
        val_losses=val_losses,
    )
    pickle.dump(report_dict, file=open(save_path / "report_dict.pkl", "wb"))
    
    
    for split in ["TRAIN", "VAL"]:
        images_dir = save_path / split
        gif_name = save_path / f"{split}_epochs.gif"
        
        create_gif_from_images(images_dir=images_dir, gif_path_name=gif_name)

    return model, report_dict
        

@torch.no_grad()
def validate_model(model, dataloader, epoch, save_path, show_plot, device, is_test:bool, 
                   torchmetrics_JaccardIndex, torchmetrics_Accuracy, criterion, use_cross_entropy):
    
    model.eval()
    
    
    running_loss = 0.0
    running_samples = 0
    
    
    iou_accuracies = []
    pixel_accuracies = []
    iou_scores = []
    split = "TEST" if is_test else "VAL"
    
    
    inputs_for_plot = None
    targets_for_plot = None
    predictions_for_plot = None

    for batch_number, (inputs, targets) in enumerate(dataloader, 1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        logits = model(inputs)        
        predictions = logits.softmax(dim=1)
        
        predicted_labels = predictions.argmax(dim=1, keepdims=True) # Has the same shape

        predicted_mask = predicted_labels.float()
        
        custom_iou_batch = IoUMetric(predictions, targets)
        iou_accuracy_batch = torchmetrics_JaccardIndex(predicted_mask, targets)
        pixel_accuracy_batch = torchmetrics_Accuracy(predicted_labels, targets)

        
        if batch_number == 1:
            inputs_for_plot = inputs
            targets_for_plot = targets
            predictions_for_plot = predicted_mask

        if use_cross_entropy:
            targets_criterion = targets.squeeze(1)
        else:
            targets_criterion = targets_criterion
            
        loss = criterion(logits, targets_criterion)

        # iou = to_device()
        # pixel_metric = to_device()
        
        
        
        iou_accuracies.append(iou_accuracy_batch)
        pixel_accuracies.append(pixel_accuracy_batch)
        iou_scores.append(custom_iou_batch)
        running_loss += loss.item()
        
        
    pixel_accuracy = torch.FloatTensor(pixel_accuracies).mean()
    iou_accuracy = torch.FloatTensor(iou_accuracies).mean()
    custom_iou = torch.FloatTensor(iou_scores).mean()
    
    total_loss = running_loss / batch_number
    

    title = f'[{split}] Epoch: {epoch}, Loss: {total_loss:.4f}, Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Custom IoU: {custom_iou:.4f}]'
    print(title)
    # print(f"Accuracy: {accuracy:.4f}")
    plot_inputs_targets_predictions(inputs=inputs_for_plot, targets=targets_for_plot, predictions=predictions_for_plot, 
                                        save_path=str(save_path / split / f"epoch_{epoch}.png"),
                                        title=title, show_plot=False)
    

    

        
        
    return title, pixel_accuracy, iou_accuracy, custom_iou, total_loss



def test_dataset_accuracy(model, loader):
    to_device(model.eval())
    iou = to_device(TM.classification.MulticlassJaccardIndex(2, average='micro', ignore_index=BimapClasses.BACKGROUND))
    pixel_metric = to_device(TM.classification.MulticlassAccuracy(2, average='micro'))
    
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

        # print("pred_labels.shape: {}".format(pred_labels.shape))
        pred_mask = pred_labels.float()

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