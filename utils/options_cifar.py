
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=7, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--client_optimizer', type=str, default='adam', help='SGD with momentum; adam')
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--aggregation_methods', type=str, default="fedavg",
                        help='fedavg/krum/trimmed_mean')
    parser.add_argument('--lr_decay', type=float, default=0.99)

    # attack
    parser.add_argument('--comprised_rate', type=float, default=0.2)
    parser.add_argument('--attack_type', type=str, default="MSA",
                        help='MSA,gaussian_attack,no_attack')
    parser.add_argument('--shuffle_ratio', type=float, default=1,
                        help='0.25,0.5,0.75,1')

    # dp
    parser.add_argument('--use_dp', type=bool, default=True,
                        help='True/False')
    parser.add_argument('--clipthr', type=int, default=40,
                        help='threshold for parameter pruning')
    parser.add_argument('--privacy_budget', type=int, default=40)
    parser.add_argument('--attackers_privacy_budget', type=int, default=100)
    parser.add_argument('--delta', type=float, default=0.01)

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='None', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')


    args = parser.parse_args()
    return args
