import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import time
import random
import numpy as np
import yaml
import os
import math
from transformer_model import TransformerModel
from data_processing import load_and_preprocess_data, get_data_loaders
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
#Code below just sets up the logging for the training process
def setup_logging(config_name: str) -> None:
    """
    Set up logging for the training process.
    
    Args:
        config_name (str): Name of the configuration file (used for log file naming).
    """
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{config_name}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
#Function for one epoch of training to be done on the model
def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                train_loader: torch.utils.data.DataLoader, device: torch.device, 
                use_mixed_precision: bool, scaler: GradScaler, clip_grad_norm: float, 
                writer: SummaryWriter, epoch: int, total_steps: int, scheduler) -> tuple[float, int]:
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The Transformer model to train.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        criterion (nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        device (torch.device): The device to use for training.
        use_mixed_precision (bool): Whether to use mixed precision training.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        clip_grad_norm (float): Maximum norm for gradient clipping.
        writer (SummaryWriter): TensorBoard summary writer.
        epoch (int): Current epoch number.
        total_steps (int): Total number of training steps so far.
    
    Returns:
        tuple[float, int]: Average loss for the epoch and updated total steps.
    """
    #set to training mode
    model.train()
    #initialize the total loss and total elements
    total_loss = 0.
    start_time = time.time()
    #Progress bar for visual feedback
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
    #Iterate over the batches in the training data
    for batch, (data, targets) in enumerate(progress_bar):
        #choose device for training
        data, targets = data.to(device), targets.to(device)
        #zero the gradients
        optimizer.zero_grad()
        #If mixed precision is used, apply autocast to the model. Used to speed up the model training
        if use_mixed_precision:
            with autocast(device_type="cuda"):
                #resul tof one forward pass
                output = model(data)
                #Calculate the loss
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            #Backpropagate the loss
            scaler.scale(loss).backward()
            #If gradient clipping is used, clip the gradients, prevents explosion of gradients
            if clip_grad_norm > 0.0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            #Step the optimizer using the scaled gradients
            scaler.step(optimizer)
            #Update the scaler
            scaler.update()
        else:
            #If mixed precision is not used, apply the forward pass, calculate the loss and backpropagate
            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
        #Update the total loss and current loss
        total_loss += loss.item()
        current_loss = loss.item()
        #Calc the current bpc 
        current_bpc = loss.item() / math.log(2)

        current_time = time.time() - start_time

        if (batch + total_steps) % 100 == 0:
            #Below using tensorboard to track progress and track various factors
            writer.add_scalar('Loss/train_step', loss.item(), total_steps + batch)
            writer.add_scalar('BPC/train_step', loss.item() / math.log(2), total_steps + batch)
            writer.add_scalar('Time/step', current_time / (batch + 1), total_steps + batch)
        #Update the learning rate using the scheduler
        scheduler.step()
        #Update the progress bar with the current loss, bpc and time per batch
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'bpc': f'{current_bpc:.4f}',
            'time/batch': f'{current_time / (batch + 1):.3f}s'
        })
    #Return the average loss for the epoch and the total steps
    return total_loss / len(train_loader), total_steps + len(train_loader)
#Function to evaluate the model on the validation or test data
def evaluate(model: nn.Module, criterion: nn.Module, val_loader: torch.utils.data.DataLoader, 
             device: torch.device, writer: SummaryWriter, epoch: int, step: int, prefix: str = 'val') -> float:
    """
    Evaluate the model on validation or test data.
    
    Args:
        model (nn.Module): The Transformer model to evaluate.
        criterion (nn.Module): The loss function.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation/test data.
        device (torch.device): The device to use for evaluation.
        writer (SummaryWriter): TensorBoard summary writer.
        epoch (int): Current epoch number.
        step (int): Current step number.
        prefix (str): Prefix for logging ('val' or 'test').
    
    Returns:
        float: Average loss for the evaluation.
    """
    #Set the model to evaluation mode
    model.eval()
    #Initialize the total loss
    total_loss = 0.
    start_time = time.time()
    progress_bar = tqdm(val_loader, desc=f"Evaluating {prefix.capitalize()}", leave=False)
    #Iterate over the batches in the dataset
    #(Not doing to repeat comments... same as above, just with no backprop or training steps  )
    
    with torch.no_grad():
        for batch, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            total_loss += loss.item()
            #Calculate the current loss, bpc and time per batch
            current_loss = total_loss / (batch + 1)
            current_bpc = loss.item() / math.log(2)
            current_time = time.time() - start_time
            #Log the metrics to tensorboard every 100 steps
            if (batch + step) % 100 == 0:
                writer.add_scalar(f'Loss/{prefix}_step', loss, step + batch)
                writer.add_scalar(f'BPC/{prefix}_step', loss / math.log(2), step + batch)
                writer.add_scalar(f'Time/{prefix}_step', current_time / (batch + 1), step + batch)
            #Update the progress bar with the current loss, bpc and time per batch
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'bpc': f'{current_bpc:.4f}',
                'time/batch': f'{current_time / (batch + 1):.3f}s'
            })
    #Return the average loss for the evaluated dataset partition
    return total_loss / len(val_loader)
#Function to train the model according to the various parameters specified in our configs
def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, 
                num_epochs: int, device: torch.device, use_mixed_precision: bool, 
                clip_grad_norm: float, config_name: str) -> None:
    """
    Train the Transformer model.
    
    Args:
        model (nn.Module): The Transformer model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
        device (torch.device): The device to use for training.
        use_mixed_precision (bool): Whether to use mixed precision training.
        clip_grad_norm (float): Maximum norm for gradient clipping.
        config_name (str): Name of the configuration file.
        tokenizer: The tokenizer used for the data.
    """
    #set below accrding to config parameters
    scaler = GradScaler(device=device) if use_mixed_precision else None
    
    writer = SummaryWriter(f'runs/{config_name}')
    #initialize the best validation loss to infinity
    best_val_loss = float('inf')
    total_steps = 0
    #Iterate over the number of epochs specified in the config
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        #Call the train epoch function to train the model
        train_loss, total_steps = train_epoch(model, optimizer, criterion, train_loader, device, use_mixed_precision, scaler, clip_grad_norm, writer, epoch, total_steps, scheduler)
        #Call the evaluate function to evaluate the model on the validation data
        val_loss = evaluate(model, criterion, val_loader, device, writer, epoch, total_steps, prefix='val')
        #Calculate the time taken for the epoch and the bpc for the training and validation data
        epoch_time = time.time() - epoch_start_time
        train_bpc = train_loss / math.log(2)
        val_bpc = val_loss / math.log(2)
        
        #BElow tracking info for tensorboard
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('BPC/train_epoch', train_bpc, epoch)
        writer.add_scalar('BPC/val_epoch', val_bpc, epoch)
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        
        log_message = f'| Epoch {epoch:3d} | time: {epoch_time:.4f}s | train loss {train_loss:.4f} | 'f'train bpc {train_bpc:.4f} | valid loss {val_loss:.4f} | valid bpc {val_bpc:.4f}'
        logging.info(log_message)
        print(log_message)
        #Keeping track of the best validation loss and saving the model if the current validation loss is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{config_name}_best_model.pt')

    writer.close()
    logging.info("Training completed.")
#Function to set the seed for reproducibility
def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
#Function to load the config file
def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Loaded configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
#Function to get the scheduler for the optimizer, based on one specified in the config file
def get_scheduler(scheduler_type: str, optimizer: torch.optim.Optimizer, 
                  num_warmup_steps: int, num_training_steps: int, args: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get the appropriate learning rate scheduler.
    
    Args:
        scheduler_type (str): Type of scheduler to use.
        optimizer (torch.optim.Optimizer): The optimizer for model parameters.
        num_warmup_steps (int): Number of warmup steps for schedulers that use warmup.
        num_training_steps (int): Total number of training steps.
        args (dict): Additional arguments for the scheduler.
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: The selected learning rate scheduler.
    """
    if scheduler_type == 'linear_warmup_decay':
        #Below sets up the linear warmup decay scheduler
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_type == 'cosine_warmup_decay':
        #Below sets up the cosine warmup decay scheduler
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=args.get('lr_step_size', 1), gamma=args.get('lr_gamma', 0.95))
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
#implmentation of the linear warmup decay scheduler
def get_linear_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        warmup_lr (float): The initial learning rate for the warmup phase.
    
    Returns:
        LambdaLR: The learning rate scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)
#Implementation of the cosine warmup decay scheduler
def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between 0
    and `pi`, after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        warmup_lr (float): The initial learning rate for the warmup phase.
    
    Returns:
        LambdaLR: The learning rate scheduler.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)
#function to rrun experiments using config files        
def run_experiment(config_path: str) -> None:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config_path (str): Path to the configuration file for the experiment.
    """
    #Set up logging
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    setup_logging(config_name)
    logging.info(f"Starting experiment with config: {config_name}")
    #Load the config file
    config = load_config(config_path)
    set_seed(config['random_seed'])
    #Set the device to cuda if available, otherwise to cpu
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    #load datasets and apply our preprocessing functions to them
    train_dataset, valid_dataset, test_dataset, tokenizer = load_and_preprocess_data(config['data']['dir'], config['data']['seq_length'])
    train_loader, valid_loader, test_loader = get_data_loaders(train_dataset, valid_dataset, test_dataset, config['data']['batch_size'])

    config['model']['vocab_size'] = tokenizer.vocab_size
    #Initialize the model, criterion and optimizer based on config fields
    model = TransformerModel(**config['model'])
    model = model.to(device)
    #Below sets up the cross entropy loss criterion and the adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    #Set up the scheduler for the optimizer based on config params
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    num_warmup_steps = int(num_training_steps * config['training'].get('warmup_ratio', 0.5))
    print(f"Num training steps: {num_training_steps}, Num warmup steps: {num_warmup_steps}")
    #Get the scheduler for the optimizer
    scheduler = get_scheduler(config['training']['lr_schedule'], optimizer, num_warmup_steps, num_training_steps, config['training'])
    #Train the model based on parameters specified in configs
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, config['training']['num_epochs'], device, config['training']['use_mixed_precision'], config['training'].get('clip_grad_norm', 1.0),config_name)
    
    model.load_state_dict(torch.load(f'{config_name}_best_model.pt'))
    test_loss = evaluate(model, criterion, test_loader, device, SummaryWriter(f'runs/{config_name}_test'), 0, 0, prefix='test')
    test_bpc = test_loss / math.log(2)
    logging.info(f'Test Loss: {test_loss:.4f}')
    logging.info(f'Test BPC: {test_bpc:.4f}')

def main() -> None:
    """
    Main function to run experiments for all configuration files in the 'configs' directory.
    """
    config_dir = 'configs'
    for config_file in os.listdir(config_dir):
        if config_file.endswith('.yaml'):
            config_path = os.path.join(config_dir, config_file)
            run_experiment(config_path)

if __name__ == "__main__":
    main()