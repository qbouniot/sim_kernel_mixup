import numpy as np
import copy
# import ipdb
import torch
import torch.nn as nn

import time
from torch.optim import Adam
from sklearn.neighbors import KernelDensity
from utils import stats_values
import scipy.stats

from netcal.metrics import NLL, QCE, PinballLoss,UCE, ENCE
from netcal.regression import VarianceScaling

def cal_worst_acc(args,data_packet,best_model_rmse,best_result_dict_rmse,all_begin,ts_data,device):
    #### worst group acc ---> rmse ####
    if args.is_ood:
        x_test_assay_list = data_packet['x_test_assay_list']
        y_test_assay_list = data_packet['y_test_assay_list']
        worst_acc = 0.0 if args.metrics == 'rmse' else 1e10
            
        for i in range(len(x_test_assay_list)):
            result_dic = test(args,best_model_rmse,x_test_assay_list[i],y_test_assay_list[i],
                            '', False, all_begin, device)
            acc = result_dic[args.metrics] 
            if args.metrics == 'rmse':
                if acc > worst_acc:
                    worst_acc = acc
            else:#r
                if np.abs(acc) < np.abs(worst_acc):
                    worst_acc = acc
        print('worst {} = {:.3f}'.format(args.metrics, worst_acc))
        best_result_dict_rmse['worst_' + args.metrics] = worst_acc

def get_mixup_sample_rate(args, data_packet, device='cuda', use_kde = False):
    
    mix_idx = []
    _, y_list = data_packet['x_train'], data_packet['y_train'] 
    is_np = isinstance(y_list,np.ndarray)
    if is_np:
        data_list = torch.tensor(y_list, dtype=torch.float32)
    else:
        data_list = y_list

    N = len(data_list)

    ######## use kde rate or uniform rate #######
    for i in range(N):
        if args.mixtype == 'kde' or use_kde: # kde
            data_i = data_list[i]
            data_i = data_i.reshape(-1,data_i.shape[0]) # get 2D
            
            if args.show_process:
                if i % (N // 10) == 0:
                    print('Mixup sample prepare {:.2f}%'.format(i * 100.0 / N ))
                # if i == 0: print(f'data_list.shape = {data_list.shape}, std(data_list) = {torch.std(data_list)}')#, data_i = {data_i}' + f'data_i.shape = {data_i.shape}')
                
            ######### get kde sample rate ##########
            kd = KernelDensity(kernel=args.kde_type, bandwidth=args.kde_bandwidth).fit(data_i)  # should be 2D
            each_rate = np.exp(kd.score_samples(data_list))
            each_rate /= np.sum(each_rate)  # norm
        else:
            each_rate = np.ones(y_list.shape[0]) * 1.0 / y_list.shape[0]
        
        ####### visualization: observe relative rate distribution shot #######
        if args.show_process and i == 0:
                print(f'bw = {args.kde_bandwidth}')
                print(f'each_rate[:10] = {each_rate[:10]}')
                stats_values(each_rate)
            
        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]

    if args. show_process:
        print(f'len(y_list) = {len(y_list)}, len(mix_idx) = {len(mix_idx)}, np.mean(self_rate) = {np.mean(self_rate)}, np.std(self_rate) = {np.std(self_rate)},  np.min(self_rate) = {np.min(self_rate)}, np.max(self_rate) = {np.max(self_rate)}')

    return mix_idx

def get_batch_kde_mixup_idx(args, Batch_X, Batch_Y, device):
    assert Batch_X.shape[0] % 2 == 0
    Batch_packet = {}
    Batch_packet['x_train'] = Batch_X.cpu()
    Batch_packet['y_train'] = Batch_Y.cpu()

    Batch_rate = get_mixup_sample_rate(args, Batch_packet, device, use_kde=True) # batch -> kde
    if args. show_process:
        stats_values(Batch_rate[0])
        # print(f'Batch_rate[0][:20] = {Batch_rate[0][:20]}')
    idx2 = [np.random.choice(np.arange(Batch_X.shape[0]), p=Batch_rate[sel_idx]) 
            for sel_idx in np.arange(Batch_X.shape[0]//2)]
    return idx2

def get_batch_kde_mixup_batch(args, Batch_X1, Batch_X2, Batch_Y1, Batch_Y2, device):
    Batch_X = torch.cat([Batch_X1, Batch_X2], dim = 0)
    Batch_Y = torch.cat([Batch_Y1, Batch_Y2], dim = 0)

    idx2 = get_batch_kde_mixup_idx(args,Batch_X,Batch_Y,device)

    New_Batch_X2 = Batch_X[idx2]
    New_Batch_Y2 = Batch_Y[idx2]
    return New_Batch_X2, New_Batch_Y2


def test(args, model, x_list, y_list, name, need_verbose, epoch_start_time, device, mc_dropout=False, samples=50, varscaling=False, x_val=None, y_val=None):
    model.eval()
    with torch.no_grad():
        if args.dataset == 'Dti_dg': 
            val_iter = x_list.shape[0] // args.batch_size 
            val_len = args.batch_size
            y_list = y_list[:val_iter * val_len]
        else: # read in the whole test data
            val_iter = 1
            val_len = x_list.shape[0]
        y_list_pred = []
        assert val_iter >= 1 #  easy test

        for ith in range(val_iter):
            if isinstance(x_list,np.ndarray):
                x_list_torch = torch.tensor(x_list[ith*val_len:(ith+1)*val_len], dtype=torch.float32).to(device)
            else:
                x_list_torch = x_list[ith*val_len:(ith+1)*val_len].to(device)

            model = model.to(device)
            if not mc_dropout:
                pred_y = model(x_list_torch).cpu().numpy()
            else:
                preds_mc = []
                for i in range(samples):
                    pred_y_mc = model(x_list_torch, mc_dropout=True).cpu().numpy()
                    preds_mc.append(pred_y_mc)
                pred_y = (np.mean(preds_mc, axis=0), np.std(preds_mc, axis=0))
            y_list_pred.append(pred_y)

        y_list_pred = np.concatenate(y_list_pred,axis=0)
        y_list = y_list.squeeze()
        y_list_pred = y_list_pred.squeeze()

        if not isinstance(y_list, np.ndarray):
            y_list = y_list.numpy()

        if varscaling and mc_dropout:
            varscaling = VarianceScaling()
            if isinstance(x_val,np.ndarray):
                x_val = torch.tensor(x_val, dtype=torch.float32, device=device)
            else:
                x_val = x_val.to(device)
            preds_valid_mc = []
            for i in range(samples):
                pred_val_y_mc = model(x_val, mc_dropout=True).cpu().numpy()
                preds_valid_mc.append(pred_val_y_mc)
            pred_y_val = (np.mean(preds_valid_mc, axis=0), np.std(preds_valid_mc, axis=0))
            varscaling.fit(pred_y_val, y_val)
            pred_std_scaled = varscaling.transform(y_list_pred)

            # varscaling.fit(y_list_pred, y_list)
            # pred_std_scaled = varscaling.transform(y_list_pred)
            y_list_pred_scaled = (y_list_pred[0], pred_std_scaled)
        
        ###### calculate metrics ######

        if not mc_dropout:
            mean_p = y_list_pred.mean(axis = 0)
            sigma_p = y_list_pred.std(axis = 0)
            mean_g = y_list.mean(axis = 0)
            sigma_g = y_list.std(axis = 0)

            index = (sigma_g!=0)
            corr = ((y_list_pred - mean_p) * (y_list - mean_g)).mean(axis = 0) / (sigma_p * sigma_g)
            corr = (corr[index]).mean()

            mse = (np.square(y_list_pred  - y_list )).mean()
            result_dict = {'mse':mse, 'r':corr, 'r^2':corr**2, 'rmse':np.sqrt(mse)}

            not_zero_idx = y_list != 0.0
            mape = (np.fabs(y_list_pred[not_zero_idx] -  y_list[not_zero_idx]) / np.fabs(y_list[not_zero_idx])).mean() * 100
            result_dict['mape'] = mape
        else:
            mean_p = y_list_pred[0].mean(axis = 0)
            sigma_p = y_list_pred[0].std(axis = 0)
            mean_g = y_list.mean(axis = 0)
            sigma_g = y_list.std(axis = 0)

            index = (sigma_g!=0)
            corr = ((y_list_pred[0] - mean_p) * (y_list - mean_g)).mean(axis = 0) / (sigma_p * sigma_g)
            corr = (corr[index]).mean()

            mse = (np.square(y_list_pred[0]  - y_list )).mean()
            result_dict = {'mse':mse, 'r':corr, 'r^2':corr**2, 'rmse':np.sqrt(mse)}

            not_zero_idx = y_list != 0.0
            mape = (np.fabs(y_list_pred[0][not_zero_idx] -  y_list[not_zero_idx]) / np.fabs(y_list[not_zero_idx])).mean() * 100
            result_dict['mape'] = mape

            nb_bins = 15
            kind = "meanstd"
            quantiles = np.linspace(0.05,0.95,19)

            nll = NLL()
            pinball = PinballLoss()
            qce = QCE(bins=nb_bins, marginal=False)
            uce = UCE(bins=nb_bins)
            ence = ENCE(bins=nb_bins)

            nll_loss = nll.measure(y_list_pred, y_list, kind=kind, reduction='mean')
            pinball_loss = pinball.measure(y_list_pred, y_list, q=quantiles, kind=kind, reduction='mean')
            qce_loss = qce.measure(y_list_pred, y_list, q=quantiles, kind=kind, reduction='mean')
            uce_loss = uce.measure(y_list_pred, y_list, kind=kind)
            ence_loss = ence.measure(y_list_pred, y_list, kind=kind)

            result_dict["nll"] = float(nll_loss)
            result_dict["pinball"] = float(pinball_loss)
            result_dict["qce"] = float(qce_loss)
            result_dict["uce"] = float(np.mean(uce_loss))
            result_dict["ence"] = float(np.mean(ence_loss))

            if varscaling:
                nll_loss_scaled = nll.measure(y_list_pred_scaled, y_list, kind=kind, reduction='mean')
                pinball_loss_scaled = pinball.measure(y_list_pred_scaled, y_list, q=quantiles, kind=kind, reduction='mean')
                qce_loss_scaled = qce.measure(y_list_pred_scaled, y_list, q=quantiles, kind=kind, reduction='mean')
                uce_loss_scaled = uce.measure(y_list_pred_scaled, y_list, kind=kind)
                ence_loss_scaled = ence.measure(y_list_pred_scaled, y_list, kind=kind)

                result_dict["nll_scaled"] = float(nll_loss_scaled)
                result_dict["pinball_scaled"] = float(pinball_loss_scaled)
                result_dict["qce_scaled"] = float(qce_loss_scaled)
                result_dict["uce_scaled"] = float(np.mean(uce_loss_scaled))
                result_dict["ence_scaled"] = float(np.mean(ence_loss_scaled))
            
        
    ### verbose ###
    if need_verbose:
        epoch_use_time = time.time() - epoch_start_time
        # valid -> interval time; final test -> all time
        print(name + 'corr = {:.4f}, rmse = {:.4f}, mape = {:.4f} %'.format(corr,np.sqrt(mse),mape) + ', time = {:.4f} s'.format(epoch_use_time))
        
    return result_dict



def train(args, model, data_packet, is_mixup=True, mixup_idx_sample_rate=None, ts_data= None, device='cuda'):
    ######### model prepare ########
    model.train(True)    
    optimizer = Adam(model.parameters(), **args.optimiser_args)
    loss_fun = nn.MSELoss(reduction='mean').to(device)
    
    best_mse = 1e10  # for best update
    best_r2 = 0.0
    repr_flag = 1 # for batch kde visualize training process

    scheduler = None
    
    x_train = data_packet['x_train']
    y_train = data_packet['y_train']
    x_valid = data_packet['x_valid']
    y_valid = data_packet['y_valid']

    iteration = len(x_train) // args.batch_size
    steps_per_epoch = iteration

    result_dict,best_mse_model = {},None
    step_print_num = 30 # for dti

    need_shuffle = not args.is_ood

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        shuffle_idx = np.random.permutation(np.arange(len(x_train)))

        if need_shuffle: # id
            x_train_input = x_train[shuffle_idx]
            y_train_input = y_train[shuffle_idx]
        else:# ood
            x_train_input = x_train
            y_train_input = y_train

        if not is_mixup:

            # iteration for each batch
            for idx in range(iteration):
                # select batch
                x_input_tmp = x_train_input[idx * args.batch_size:(idx + 1) * args.batch_size]
                y_input_tmp = y_train_input[idx * args.batch_size:(idx + 1) * args.batch_size]

                # -> tensor
                x_input = torch.tensor(x_input_tmp, dtype=torch.float32).to(device)
                y_input = torch.tensor(y_input_tmp, dtype=torch.float32).to(device)

                # forward
                pred_Y = model(x_input)
                if args.dataset == 'TimeSeries': # time series loss need scale
                    # scale = ts_data.scale.expand(pred_Y.size(0),ts_data.m)
                    scale = ts_data[0].expand(pred_Y.size(0), ts_data[1]) # ts_data = [scale, m]
                    loss = loss_fun(pred_Y * scale, y_input * scale)
                else:
                    loss = loss_fun(pred_Y, y_input)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler != None: # backward (without scheduler)
                    scheduler.step()

        else:  # mix up
            for idx in range(iteration):

                if args.mixtype == 'kernel_sim':
                    lambd = np.random.beta(args.mix_alpha, args.mix_alpha, args.batch_size)
                else:
                    lambd = np.random.beta(args.mix_alpha, args.mix_alpha)

                if need_shuffle: # get batch idx
                    idx_1 = shuffle_idx[idx * args.batch_size:(idx + 1) * args.batch_size]
                else:
                    idx_1 = np.arange(len(x_train))[idx * args.batch_size:(idx + 1) * args.batch_size]
                
                if args.mixtype == 'kde' or args.use_kde: 
                    idx_2 = np.array(
                        [np.random.choice(np.arange(x_train.shape[0]), p=mixup_idx_sample_rate[sel_idx]) for sel_idx in
                        idx_1])
                else: # random mix
                    idx_2 = np.array(
                        [np.random.choice(np.arange(x_train.shape[0])) for sel_idx in idx_1])

                if isinstance(x_train,np.ndarray):
                    X1 = torch.tensor(x_train[idx_1], dtype=torch.float32).to(device)
                    Y1 = torch.tensor(y_train[idx_1], dtype=torch.float32).to(device)

                    X2 = torch.tensor(x_train[idx_2], dtype=torch.float32).to(device)
                    Y2 = torch.tensor(y_train[idx_2], dtype=torch.float32).to(device)
                else:
                    X1 = x_train[idx_1].to(device)
                    Y1 = y_train[idx_1].to(device)

                    X2 = x_train[idx_2].to(device)
                    Y2 = y_train[idx_2].to(device)

                if args.batch_type == 1: # sample from batch
                    assert args.mixtype == 'random'
                    if not repr_flag: # show the sample status once
                        args.show_process = 0
                    else:
                        repr_flag = 0
                    X2, Y2 = get_batch_kde_mixup_batch(args,X1,X2,Y1,Y2,device)
                    args.show_process = 1

                X1 = X1.to(device)
                X2 = X2.to(device)
                Y1 = Y1.to(device)
                Y2 = Y2.to(device)

                # kernel warping mixup
                if args.mixtype == 'kernel_sim':

                    if args.input_sim_dist == "norm_cent_gauss_out_l2":
                        label_dist = (Y1 - Y2).pow(2).sum(-1).cpu().numpy()
                        label_dist /= np.mean(label_dist)
                        input_rate = np.exp(-(label_dist-1) / ( 2 * args.tau_std_x * args.tau_std_x))
                        input_rate = args.tau_max_x * input_rate
                    elif args.input_sim_dist == "norm_cent_gauss_inp_l2":
                        if args.dataset == 'TimeSeries':
                            inp_dist = (X1 - X2).pow(2).sum([-2,-1]).cpu().numpy()
                        else:
                            inp_dist = (X1 - X2).pow(2).sum(-1).cpu().numpy()
                        inp_dist /= np.mean(inp_dist)
                        
                        input_rate = np.exp(-(inp_dist-1) / ( 2 * args.tau_std_x * args.tau_std_x))
                        input_rate = args.tau_max_x * input_rate
                    elif args.input_sim_dist == "norm_cent_gauss_feat_l2":
                        with torch.no_grad():
                            feats = model.repr_forward(torch.tensor(x_train, dtype=torch.float32, device=device)).detach()
                        feat_dist = (feats[idx_1] - feats[idx_2]).pow(2).sum(-1).cpu().numpy()
                        feat_dist /= np.mean(feat_dist)
                        
                        input_rate = np.exp(-(feat_dist-1) / ( 2 * args.tau_std_x * args.tau_std_x))
                        input_rate = args.tau_max_x * input_rate

                    if args.output_sim_dist == 'norm_cent_gauss_out_l2':
                        label_dist = (Y1 - Y2).pow(2).sum(-1).cpu().numpy()
                        label_dist /= np.mean(label_dist)
        
                        output_rate = np.exp(-(label_dist-1) / (2 * args.tau_std_y * args.tau_std_y))
                        output_rate = args.tau_max_y * output_rate

                    k_lam_X = torch.tensor(beta_warping(lambd, input_rate), device=X1.device).unsqueeze(-1).float() 
                    k_lam_Y = torch.tensor(beta_warping(lambd, output_rate), device=Y1.device).unsqueeze(-1).float()

                    if args.dataset == 'TimeSeries':
                        k_lam_X.unsqueeze_(-1)
                        # k_lam_Y.unsqueeze_(-1)

                    mixup_X = X1 * k_lam_X + X2 * (1 - k_lam_X)
                    mixup_Y = Y1 * k_lam_Y + Y2 * (1 - k_lam_Y)

                else:
                    # mixup
                    mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
                    mixup_X = X1 * lambd + X2 * (1 - lambd)
                
                # forward
                if args.use_manifold:
                    if args.mixtype=='kernel_sim':
                        pred_Y = model.forward_mixup(X1, X2, k_lam_X)
                    else:
                        pred_Y = model.forward_mixup(X1, X2, lambd)
                else:
                    pred_Y = model.forward(mixup_X)

                if args.dataset == 'TimeSeries': # time series loss need scale
                    # scale = ts_data.scale.expand(pred_Y.size(0),ts_data.m)
                    scale = ts_data[0].expand(pred_Y.size(0), ts_data[1]) # ts_data = [scale, m]
                    loss = loss_fun(pred_Y * scale, mixup_Y * scale)
                else:    
                    loss = loss_fun(pred_Y, mixup_Y)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # validation
        result_dict = test(args, model, x_valid, y_valid, 'Train epoch ' + str(epoch) +':\t', args.show_process, epoch_start_time, device)
        

        # if args.is_ood:
        #     cal_worst_acc(args,data_packet,model,result_dict,epoch_start_time,ts_data,device)
        #     worst_test_loss_log.append(result_dict['worst_rmse']**2)

        if result_dict['mse'] <= best_mse:
            best_mse = result_dict['mse']
            best_mse_model = copy.deepcopy(model)
            print(f'update best mse! epoch = {epoch}')
        
        if result_dict['r']**2 >= best_r2:
            best_r2 = result_dict['r']**2
            best_r2_model = copy.deepcopy(model)

    return best_mse_model, best_r2_model

def beta_warping(x, alpha_cdf=1., eps=1e-12):

    return scipy.stats.beta.ppf(x, a=alpha_cdf+eps, b=alpha_cdf+eps)