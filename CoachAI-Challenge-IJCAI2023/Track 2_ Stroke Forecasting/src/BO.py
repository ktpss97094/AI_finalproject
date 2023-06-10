from badmintoncleaner import prepare_dataset
import os
import torch
import torch.nn as nn
import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import sys

from bayes_opt import BayesianOptimization


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(batch_size, lr, epochs, area_num, dim):
    
    # shot_dim, area_dim, player_dim, encode_dim 不一樣好像會報錯

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
        'data_folder': "../data/",
        'model_folder': './model/'
    }

    #config['max_ball_round'] = int(max_ball_round)
    #config['encode_length'] = int(encode_length)
    config['batch_size'] = int(batch_size)
    config['lr'] = lr
    config['epochs'] = int(epochs)
    config['area_num'] = int(area_num)
    config['shot_dim'] = int(dim)
    config['area_dim'] = int(dim)
    config['player_dim'] = int(dim)
    config['encode_dim'] = int(dim)

    # print(
    # config['max_ball_round'],
    # config['encode_length'],
    # config['batch_size'],
    # config['lr'],
    # config['epochs'],
    # config['shot_dim'],
    # config['area_num'],
    # config['area_dim'],
    # config['player_dim'],
    # config['encode_dim']
    # )

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

def generate():
    SAMPLES = 6 # set to 6 to meet the requirement of this challenge

    model_path = "./model" 
    config = ast.literal_eval(open(f"{model_path}/config").readline())
    set_seed(config['seed_value'])

    # Prepare Dataset
    config, train_dataloader, val_dataloader, test_dataloader, train_matches, val_matches, test_matches = prepare_dataset(config)
    device = torch.device(f"cuda:{config['gpu_num']}" if torch.cuda.is_available() else "cpu")

    # load model
    from ShuttleNet.ShuttleNet import ShotGenEncoder, ShotGenPredictor
    from ShuttleNet.ShuttleNet_runner import shotgen_generator
    encoder = ShotGenEncoder(config)
    decoder = ShotGenPredictor(config)

    encoder.to(device), decoder.to(device)
    current_model_path = model_path + '/'
    encoder_path = current_model_path + 'encoder'
    decoder_path = current_model_path + 'decoder'
    encoder.load_state_dict(torch.load(encoder_path, map_location=device)), decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encode_length = config['encode_length']

    performance_log = open(f"{current_model_path}prediction.csv", "w")
    performance_log.write("")
    performance_log.close()

    performance_log = open(f"{current_model_path}prediction.csv", "a")
    performance_log.write('rally_id,sample_id,ball_round,landing_x,landing_y,short service,net shot,lob,clear,drop,push/rush,smash,defensive shot,drive,long service')
    performance_log.write('\n')

    # get all testing rallies
    testing_rallies = val_matches['rally_id'].unique()

    for rally_id in tqdm(testing_rallies):
        # read data
        selected_matches = val_matches.loc[(val_matches['rally_id'] == rally_id)][['rally_id', 'type', 'landing_x', 'landing_y', 'player', 'rally_length', 'player_location_x', 'player_location_y']].reset_index(drop=True)
        
        generated_length = selected_matches['rally_length'][0]      # fetch the length of the current rally
        players = [selected_matches['player'][0], selected_matches['player'][1]]
        target_players = torch.tensor([players[shot_index%2] for shot_index in range(generated_length-len(selected_matches))])  # get the predicted players
        
        given_seq = {
            'given_player': torch.tensor(selected_matches['player'].values).to(device),
            'given_shot': torch.tensor(selected_matches['type'].values).to(device),
            'given_x': torch.tensor(selected_matches['landing_x'].values).to(device),
            'given_y': torch.tensor(selected_matches['landing_y'].values).to(device),
            'player_location_x': torch.tensor(selected_matches['player_location_x'].values).to(device),
            'player_location_y': torch.tensor(selected_matches['player_location_x'].values).to(device),
            'target_player': target_players.to(device),
            'rally_length': generated_length
        }

        # feed into the model
        generated_shot, generated_area = shotgen_generator(given_seq=given_seq, encoder=encoder, decoder=decoder, config=config, samples=SAMPLES, device=device)

        # store the prediction results
        for sample_id in range(len(generated_area)):
            for ball_round in range(len(generated_area[0])):
                performance_log.write(f"{rally_id},{sample_id},{ball_round+config['encode_length']+1},{generated_area[sample_id][ball_round][0]:.6f},{generated_area[sample_id][ball_round][1]:.6f},")
                for shot_id, shot_type_logits in enumerate(generated_shot[sample_id][ball_round]):
                    performance_log.write(f"{shot_type_logits:.6f}")
                    if shot_id != len(generated_shot[sample_id][ball_round]) - 1:
                        performance_log.write(",")
                performance_log.write("\n")

class evaluation:
    def __init__(self):
        self.path = "./model/"
        self.type_list = ['short service', 'net shot', 'lob', 'clear', 'drop', 'push/rush', 'smash', 'defensive shot', 'drive', 'long service']
        self.prediction = pd.read_csv(f"{self.path}prediction.csv")
        self.ground_truth = pd.read_csv(f"{self.path}val_gt.csv")

        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.ce_loss = torch.nn.NLLLoss(reduction='mean')   # we use NLL since we need to check if we need to softmax the probs of each shot

        self.check_and_convert_type_prob_with_softmax()

    def softmax(self, prob_list):
        return np.exp(prob_list) / sum(np.exp(prob_list))
    
    def check_and_convert_type_prob_with_softmax(self):
        # normalized prob of types if the sum of prob is not 1
        converted_type_probs = []
        for n, row in self.prediction.iterrows():
            # round to 5 decimals to prevent minor computation error
            
            if round(self.prediction.iloc[n][self.type_list].sum(), 5) != 1:
                converted_type_probs.append(self.softmax(row[self.type_list].values))
            else:
                converted_type_probs.append(row[self.type_list].values)
        self.prediction.loc[:, self.type_list] = converted_type_probs

    def compute_metrics(self):
        """
        for each rally
            for each sample
                compute metrics
                check if metrics is the best
        """

        total_score, total_ce_score, total_mae_score = 0, 0, 0
        rally_ids, rally_score, rally_ce_score, rally_mae_score = [], [], [], []
        group = self.prediction[['rally_id', 'sample_id', 'ball_round', 'landing_x', 'landing_y', 'short service', 'net shot', 'lob', 'clear', 'drop', 'push/rush', 'smash', 'defensive shot', 'drive', 'long service']].groupby('rally_id').apply(lambda r: (r['sample_id'].values, r['ball_round'].values, r['landing_x'].values, r['landing_y'].values, r['short service'].values, r['net shot'].values, r['lob'].values, r['clear'].values, r['drop'].values, r['push/rush'].values, r['smash'].values, r['defensive shot'].values, r['drive'].values, r['long service'].values))
        ground_truth = self.ground_truth[['rally_id', 'ball_round', 'landing_x', 'landing_y', 'type']].groupby('rally_id').apply(lambda r: (r['ball_round'].values, r['landing_x'].values, r['landing_y'].values, r['type'].values))

        for i, rally_id in tqdm(enumerate(ground_truth.index), total=len(ground_truth)):
            best_sample_score, best_ce_score, best_mae_score = 1e6, 1e6, 1e6
            sample_id, ball_round, landing_x, landing_y, short_service, net_shot, lob, clear, drop, push_rush, smash, defensive_shot, drive, long_service = group[rally_id]
            
            true_ball_round, true_landing_x, true_landing_y, true_types = ground_truth[rally_id]
            converted_true_types = []
            for true_type in true_types:
                converted_true_types.append(self.type_list.index(true_type))
            converted_true_types = torch.tensor(converted_true_types)
            ground_truth_len = len(true_ball_round)
            

            for sample in range(6):
                start_index = sample * ground_truth_len
                # compute area score
                pre_landing_x = landing_x[start_index:start_index+ground_truth_len]
                pre_landing_y = landing_y[start_index:start_index+ground_truth_len]

                if (len(pre_landing_x) == len(true_landing_x)):
                    area_score = self.compute_mae_metric(
                        pre_landing_x,
                        pre_landing_y,
                        true_landing_x,
                        true_landing_y
                    )

                # compute type score
                    prediction_type = []
                    for shot_index in range(start_index, start_index+ground_truth_len):
                        prediction_type.append([short_service[shot_index], net_shot[shot_index], lob[shot_index], clear[shot_index], drop[shot_index], push_rush[shot_index], smash[shot_index], defensive_shot[shot_index], drive[shot_index], long_service[shot_index]])
                    prediction_type = torch.tensor(prediction_type)
                    type_score = self.ce_loss(torch.log(prediction_type), converted_true_types).item()  # need to perform log operation
                    if math.isinf(type_score):
                        type_score = 1e3        # modify type_score to 1000 if the prediction prob is uniform, which causes inf

                    # check if the current score better than the previous best score
                    if area_score + type_score < best_sample_score:
                        best_sample_score = area_score + type_score
                        best_ce_score = type_score
                        best_mae_score = area_score

            rally_ids.append(rally_id), rally_score.append(best_sample_score), rally_ce_score.append(best_ce_score), rally_mae_score.append(best_mae_score)
            total_score += best_sample_score
            total_ce_score += best_ce_score
            total_mae_score += best_mae_score

        return round(total_score/len(group.index), 5)
        
    
    def compute_mae_metric(self, landing_x, landing_y, true_landing_x, true_landing_y):

        prediction_area = torch.tensor([landing_x, landing_y]).T
        true_area = torch.tensor([true_landing_x, true_landing_y]).T
        area_score = self.l1_loss(prediction_area, true_area)
        return area_score.item()
    
def BO_function(batch_size, lr, epochs, area_num, dim):

    train(batch_size, lr, epochs, area_num, dim)
    generate()
    tmp = evaluation()
    score = (1/tmp.compute_metrics()) * 100
    # 因為越小越好 所以取倒數
    # 怕因為精度問題 所以 * 100

    return score 


if __name__ == "__main__":
    
    n_iter = int(sys.argv[1]) # 總共會跑 n_iter + 5 次

    pbounds = { 
        # [最低範圍, 最高範圍] 可以自己改範圍
        # 有想要再自己加參數也可以 那 BO_function 跟 train() 也要一起改
        
        #'max_ball_round': [69, 70],
        #'encode_length': [3, 4],
        'batch_size': [30, 128],
        'lr': [1e-7, 1e-3],
        'epochs': [128, 256],
        'area_num': [5, 10], # 一定要>5
        'dim': [32, 64]
    }

    shuttle_optimizer = BayesianOptimization(f = BO_function, pbounds = pbounds)
    shuttle_optimizer.maximize(n_iter = n_iter)

    print("best hyperpara: \n", shuttle_optimizer.max)
    # 最後要跑train.py的時候把除了 lr 以外的都取整數
    # shuttle_optimizer.max 最好的hyperparameter
