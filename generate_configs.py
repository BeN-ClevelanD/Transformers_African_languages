import yaml
import os
import copy

def generate_base_config():
    return {
        #Data configuration
        'data': {
            'dir': 'LMDatasets',
            'seq_length': 64,
            'batch_size': 512
        },
        #Archiecture of model
        'model': {
            'vocab_size': 256,
            'd_model': 64,
            'nhead': 1,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'layer_norm_position': 'after',
            'weight_tying': False,
            'positional_encoding': 'learnt'
        },
        #Training configuration
        'training': {
            'weight_decay': 0.001,
            'num_epochs': 5,
            'learning_rate': 1e-6,
            'lr_schedule': 'step',
            'lr_step_size': 1,
            'lr_gamma': 0.95,
            'use_mixed_precision': True,
            'clip_grad_norm': 1.0
        },
        #Device configuration
        'device': 'cuda',
        'random_seed': 42
    }
#Function that generates the different configurations for the model based on the parameters we are varying and testing
def generate_configs():
    base_config = generate_base_config()
    configs = []
    # Default configuration
    config = copy.deepcopy(base_config)
    configs.append((f"default", config))
    
    # 1. Dropout regularization
    for dropout in [0.0]:
        config = copy.deepcopy(base_config)
        config['model']['dropout'] = dropout
        configs.append((f"dropout_{dropout}", config))

    # 2. Layer normalization position
    for position in ['before']:
        config = copy.deepcopy(base_config)
        config['model']['layer_norm_position'] = position
        configs.append((f"layernorm_{position}", config))

    # 3. Weight tying
    for weight_tying in [True]:
        config = copy.deepcopy(base_config)
        config['model']['weight_tying'] = weight_tying
        configs.append((f"weighttying_{'on' if weight_tying else 'off'}", config))

    # 4. Positional encoding
    for pos_encoding in ['fixed']:
        config = copy.deepcopy(base_config)
        config['model']['positional_encoding'] = pos_encoding
        configs.append((f"posenc_{pos_encoding}", config))

    # 5. Learning rate schedule
    for schedule in ['cosine_warmup_decay', 'linear_warmup_decay']:
        config = copy.deepcopy(base_config)
        config['training']['lr_schedule'] = schedule
        config['training']['warmup_ratio'] = 0.5
        configs.append((f"lr_schedule_{schedule}", config))
        
    # 6. Parameter tuning 
    for d_model in [64, 128]:
        for nhead in [1, 2]:
            for num_layers in [1, 2]:
                for learning_rate in [0.00001, 0.000001]:
                    for batch_size in [512, 1024]:
                        config = copy.deepcopy(base_config)
                        config['model']['d_model'] = d_model
                        config['model']['nhead'] = nhead
                        config['model']['num_layers'] = num_layers
                        config['training']['learning_rate'] = learning_rate
                        config['data']['batch_size'] = batch_size
                        configs.append((f"d_{d_model}_h_{nhead}_l_{num_layers}_lr_{learning_rate}_bs_{batch_size}", config))
    return configs
#Function that saves the configurations to a yaml file
def save_configs(configs):
    if not os.path.exists('configs'):
        os.makedirs('configs')

    for name, config in configs:
        filename = f"{name}.yaml"
        with open(os.path.join('configs', filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Generated {filename}")

if __name__ == "__main__":
    configs = generate_configs()
    save_configs(configs)