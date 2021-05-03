from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
from data.sector_data_parser import sector_data_parser
import torch

target_stocks = [
    "VZ",
    "T",
    "WMT",
    "MGM",
    "GPS",
    "GT",
    "BBY",
    "AFG",
    "ERJ",
    "MYE",
    "ECPG",
    "GCO",
    "MPC",
    "TRI",
    "UFI",
]

file_map = {}

for s in target_stocks:
    args = dotdict()

    args.model = "informer"  # model of experiment, options: [informer, informerstack, informerlight(TBD)]

    args.data = s  # data
    args.root_path = "./complete_data/"  # root path of data file
    args.data_path = "master.csv"  # data file
    args.features = "S"  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = s  # target feature in S or MS task
    args.freq = "d"  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
    args.detail_freq = args.freq
    args.checkpoints = "./15stocks_checkpoints"  # location of model checkpoints
    args.results_path = "./15stocks_results/"  # location of results

    args.seq_len = 130  # input sequence length of Informer encoder
    args.label_len = 3  # start token length of Informer decoder
    args.pred_len = 5  # prediction sequence length
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    args.enc_in = 1  # encoder input size
    args.dec_in = 1  # decoder input size
    args.c_out = 1  # output size
    args.factor = 8  # probsparse attn factor
    args.d_model = 512  # dimension of model
    args.n_heads = 8  # num of heads
    args.e_layers = 2  # num of encoder layers
    args.d_layers = 1  # num of decoder layers
    # args.s_layers = [3, 2, 1] # num of stack encoder layers?
    args.d_ff = 2048  # dimension of fcn in model
    args.dropout = 0.05  # dropout
    args.attn = "prob"  # attention used in encoder, options:[prob, full]
    args.embed = "timeF"  # time features encoding, options:[timeF, fixed, learned]
    args.activation = "gelu"  # activation
    args.distil = True  # whether to use distilling in encoder
    args.output_attention = False  # whether to output attention in ecoder

    args.batch_size = 32
    args.learning_rate = 0.0001
    args.loss = "mse"
    args.lradj = "type1"
    args.use_amp = False  # whether to use automatic mixed precision training

    args.num_workers = 0
    args.itr = 10
    args.train_epochs = 15
    args.patience = 3
    args.des = "exp"

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    args.use_multi_gpu = False
    args.devices = "0,1,2,3"

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print(f"Trainging for {s}")
    print(args)

    Exp = Exp_Informer

    best_score = 100
    best_setting = ""
    for ii in range(args.itr):
        # setting record of experiments
        setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.attn,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        # set experiments
        exp = Exp(args)

        # train
        print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
        _, score = exp.train(setting)
        if score < best_score:
            best_score = score
            best_setting = setting
        print(f"======= BEST: {ii}: {best_score} =======")

        # test
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()

    file_map[s] = {"setting": best_setting, "score": best_score}
    print(file_map)
