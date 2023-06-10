from badmintoncleaner import prepare_dataset
from utils import draw_loss
import argparse
import os
import torch
import torch.nn as nn

from bayes_opt import BayesianOptimization
import train


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(max_ball_round, encode_length, batch_size, lr, epochs, shot_dim, area_num, area_dim, player_dim, encode_dim):
    config = {
        "model_type": 'ShuttleNet',
        "output_folder_name": "./model",
        "seed_value": 42,
        'max_ball_round': 70,
        'encode_length': 4,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 150,
        'n_layers': 1,
        'shot_dim': 32,
        'area_num': 5,
        'area_dim': 32,
        'player_dim': 32,
        'encode_dim': 32,
        'num_directions': 1, # only for LSTM
        'K': 5, # fold for dataset 應該用不到
        'sample': 10, # Number of samples for evaluation 應該用不到
        'gpu_num': 0,  # Selected GPU number
        'data_folder': "../data",
        'model_folder': './model/'
    }    

    model_type = config['model_type']
    set_seed(config['seed_value'])

    # Clean data and Prepare dataset
    config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches = prepare_dataset(config)

    device = torch.device(f"cuda:{config['gpu_num']}" if torch.cuda.is_available() else "cpu")
    print("Model path: {}".format(config['output_folder_name']))
    if not os.path.exists(config['output_folder_name']):
        os.makedirs(config['output_folder_name'])

    # read model
    from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
    from ShuttleNet.ShuttleNet_runner import shotGen_trainer
    encoder = ShotGenEncoder(config)
    decoder = ShotGenPredictor(config)
    encoder.area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
    encoder.player_area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
    decoder.shotgen_decoder.player_area_embedding.weight = decoder.shotgen_decoder.area_embedding.weight
    encoder.shot_embedding.weight = decoder.shotgen_decoder.shot_embedding.weight
    encoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight
    decoder.player_embedding.weight = decoder.shotgen_decoder.player_embedding.weight

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config['lr'])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config['lr'])
    encoder.to(device), decoder.to(device)

    criterion = {
        'entropy': nn.CrossEntropyLoss(ignore_index=0, reduction='sum'),
        'mae': nn.L1Loss(reduction='sum')
    }
    for key, value in criterion.items():
        criterion[key].to(device)

    record_train_loss = shotGen_trainer(data_loader=train_dataloader, encoder=encoder, decoder=decoder, criterion=criterion, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, config=config, device=device)
    draw_loss(record_train_loss, config)