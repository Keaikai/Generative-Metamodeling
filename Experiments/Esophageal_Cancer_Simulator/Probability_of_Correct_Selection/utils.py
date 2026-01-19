import torch
import torch.optim as optim
import numpy as np
from net_utils import *
import os


def get_args(data_type):
    args = {'data_set': None,
            'data_type': data_type,
            'data_size': 10000,
            'network': 'mlp',
            'output_dim': 1,
            'latent_dim': 1,
            'num_epochs': 500,
            'test_freq': 100,
            'batch_dim': 1000,
            'hidden_dim': 128,
            'num_layer': 3,
            'output_act': None,
            'learning_rate': 1e-3,
            'learning_rate_decay': [1000,0.9],
            'weight_decay': 1e-6,
            'num_cluster': 3,
            'update_generator_freq': 5,
            'output_norm': False,
            'test_dim': 512,
            'time_step': 1000,
            'inf_step': 100,
            'eta':0.5,
            'ode_solver': 'Euler'}
    return args


# def get_args(data_type):
#     args = {'data_set': None,
#             'data_type': data_type,
#             'data_size': 10000,
#             'network': 'mlp',
#             'output_dim': 1,
#             'latent_dim': 1,
#             'num_iteration': 1000,
#             'test_freq': 100,
#             'batch_dim': 1000,
#             'hidden_dim': 128,
#             'num_layer': 3,
#             'output_act': None,
#             'learning_rate': 1e-3,
#             'learning_rate_decay': [1000,0.9],
#             'weight_decay': 1e-6,
#             'num_cluster': 3,
#             'update_generator_freq': 5,
#             'output_norm': False,
#             'test_dim': 512,
#             'time_step': 1000,
#             'inf_step': 100,
#             'eta':0.5,
#             'ode_solver': 'Euler'}
#     return args


def train_all(data, args, model_type):
    # Unpack arguments
    input_dim = data.x_train.shape[-1]
    output_dim = data.y_train.shape[1]
    data_dim = data.x_train.shape[0]
    # Set training parameters
    instance = args['instance'] 
    
    num_epochs = args['num_epochs']
    #num_epochs = args['num_iteration']
    
    batch_dim = args['batch_dim']
    hidden_dim = args['hidden_dim']
    num_layers = args['num_layer']
    latent_dim = args['latent_dim']
    output_act = args['output_act']
    network = args['network']
    pred_type = 'edge' if args['data_set'] == 'tsp' else 'node'
    # Additional parameters for different models

    time_step = args['time_step']
    output_norm = args['output_norm']
    noise_type = 'gaussian'

    if model_type == 'diffusion':
        model = DM(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
    elif model_type == 'rectified':
        model = FM(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type).to(data.device)
    else:
        raise NotImplementedError
    
    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['learning_rate_decay'][0], gamma=args['learning_rate_decay'][1])
    # Train the model
    loss_record = []
#     for epoch in range(num_epochs):
#         batch_indices = np.random.choice(data_dim, batch_dim)
#         x_batch = data.x_train[batch_indices].to(data.device)
#         y_batch = data.y_train[batch_indices].to(data.device)
#         t_batch = torch.rand([batch_dim, 1]).to(data.device)
#         if noise_type == 'gaussian':
#             z_batch = torch.randn_like(y_batch).to(data.device)
#         elif noise_type == 'uniform_fixed':
#             z_batch = torch.rand_like(y_batch).to(data.device) * 2 - 1
#         else:
#             NotImplementedError
#         optimizer.zero_grad()

#         if model_type == 'diffusion':
#             noise_pred = model.predict_noise(x_batch, y_batch, t_batch, z_batch)
#             loss = model.loss(z_batch, noise_pred)
#         elif model_type == 'rectified':
#             z_batch = torch.randn_like(y_batch).to(data.device)
#             yt, vec_target = model.flow_forward(y_batch, t_batch, z_batch, model_type)
#             vec_pred = model.predict_vec(x_batch, yt, t_batch)
#             loss = model.loss(y_batch, z_batch, vec_pred, vec_target, model_type)
#         elif model_type == 'gan':
#             y_pred =  model(x_batch, z_batch)
#             loss = model.loss_d(x_batch, y_batch, y_pred)
#             if epoch % args['update_generator_freq'] == 0:
#                 loss += model.loss_g(x_batch, y_pred)
#         else:
#             raise NotImplementedError
        
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         loss_record.append(loss.item())    

    steps_per_epoch = data_dim // batch_dim  
    for ep in range(num_epochs):
        perm = np.random.permutation(data_dim)
        for s in range(steps_per_epoch):
            batch_indices = perm[s*batch_dim:(s+1)*batch_dim]
            x_batch = data.x_train[batch_indices].to(data.device)
            y_batch = data.y_train[batch_indices].to(data.device)
            t_batch = torch.rand([batch_dim, 1]).to(data.device)
            if noise_type == 'gaussian':
                z_batch = torch.randn_like(y_batch).to(data.device)
            elif noise_type == 'uniform_fixed':
                z_batch = torch.rand_like(y_batch).to(data.device) * 2 - 1
            else:
                NotImplementedError
            optimizer.zero_grad()

            if model_type == 'diffusion':
                noise_pred = model.predict_noise(x_batch, y_batch, t_batch, z_batch)
                loss = model.loss(z_batch, noise_pred)
            elif model_type == 'rectified':
                z_batch = torch.randn_like(y_batch).to(data.device)
                yt, vec_target = model.flow_forward(y_batch, t_batch, z_batch, model_type)
                vec_pred = model.predict_vec(x_batch, yt, t_batch)
                loss = model.loss(y_batch, z_batch, vec_pred, vec_target, model_type)
            else:
                raise NotImplementedError
        
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_record.append(loss.item())
    instance = args['instance']
    torch.save(model, f'models/{instance}/{model_type}_{network}.pth')
        
    

def generate(x_test, data, args, model_name, sample_num=1):
    """
    Input：x_test: numpy (N,3) 
    Output：y_gen: numpy (N,) （denormalized）
    """
    
    # Normalize X
    x_norm = (x_test.astype(np.float32) - data.x_mean) / data.x_std
    x_input = torch.as_tensor(x_norm, dtype=torch.float32, device=data.device)
    
    instance = args['instance'] 
    inf_step = args['inf_step']
    network = args['network']
    
    test_dim = x_test.shape[0]
    
    model = torch.load(f'models/{instance}/{model_name}_{network}.pth', map_location=data.device)
    model.eval()
    
    
    if sample_num > 1:
        x_input = torch.repeat_interleave(x_input, repeats=sample_num, dim=0)

    z_test = torch.randn(size=[x_input.shape[0], args['output_dim']], device=data.device)
    
    with torch.no_grad():
        if model_name =='rectified':
            y_pred = model.flow_backward(x_input, z_test, 1/inf_step, method = args['ode_solver'])
        elif model_name == 'diffusion':
            y_pred = model.diffusion_backward(x_input, z_test, inf_step, eta=args['eta'])
        else:
            raise NotImplementedError
    
            
    if sample_num>1:
        y_pred = y_pred.view(test_dim, y_pred.shape[-1], -1) ##  batch * output_dim * sample
        
    # Denormalize
    y_np = y_pred.detach().cpu().numpy().reshape(-1)
    y_np = y_np * data.y_std + data.y_mean
    return y_np

def ensure_instance_dirs(instance):
    os.makedirs(f"models/{instance}", exist_ok=True)
    #os.makedirs(f"results/{instance}", exist_ok=True)

