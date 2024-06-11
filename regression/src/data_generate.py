
import data.airfoil as airfoil
import data.time_series as ts

def load_data(args, device):
    data_packet, ts_data = None, None
    scale, m = None, None
    if args.dataset == 'Airfoil':
        assert args.is_ood == 0
        data_packet = airfoil.get_Airfoil_data_packet(args,args.data_path / args.dataset)
    elif args.dataset == 'TimeSeries':
        assert args.is_ood == 0
        # use_cuda = 'cuda' in device

        ts_data = ts.Data_utility(args.data_dir, 0.6, 0.2, 
                            device, args.horizon, args.window, args.normalize) #0.6 0.2
    
        scale = ts_data.scale
        m = ts_data.m

        data_packet = ts.get_TimeSeries_data_packet(args, ts_data, device)

    
    if args.show_setting and not args.use_cv:
        print(
            f"x.tr,va,te; y.tr,va,te.shape = {data_packet['x_train'].shape, data_packet['x_valid'].shape, data_packet['x_test'].shape, data_packet['y_train'].shape, data_packet['y_valid'].shape, data_packet['y_test'].shape}"+
            f"y.tr.mean = {data_packet['y_train'].mean()}, y.tr.std = {data_packet['y_train'].std()}")

    return data_packet, [scale, m]

