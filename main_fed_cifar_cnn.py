import copy
import logging
import random
import time

import numpy as np
import torch
import wandb
from torchvision import datasets, transforms

from aggreagation_method.fed_aggregation import aggregation
from dp.noise_add import clipping, noise_add
from dp.privacy import privacy_account
from models.Nets import CNNCifar
from models.Update import LocalUpdate
from models.test import test_img
from utils.options_cifar import args_parser
from utils.sampling import cifar


def attack_method(w, scare_conv, scare_fc, index_conv, index_fc):
    if args.attack_type == "MSA":
        w_attacker = copy.deepcopy(w)

        # attacking
        for i in range(64):
            w_attacker['conv1.weight'][i] = w_attacker['conv1.weight'][i] * scare_conv[i]
            w_attacker['conv1.bias'][i] = w_attacker['conv1.bias'][i] * scare_conv[i]
        for k in range(128):
            for i in range(64):
                w_attacker['conv2.weight'][k][i] = w_attacker['conv2.weight'][k][i] / scare_conv[i]

        for i in range(256):
            w_attacker['fc2.weight'][i] = w_attacker['fc2.weight'][i] * scare_fc[i]
            w_attacker['fc2.bias'][i] = w_attacker['fc2.bias'][i] * scare_fc[i]
        for k in range(10):
            for i in range(256):
                w_attacker['fc3.weight'][k][i] = w_attacker['fc3.weight'][k][i] / scare_fc[i]

        w_attacker['conv1.weight'] = w_attacker['conv1.weight'][index_conv]
        w_attacker['conv1.bias'] = w_attacker['conv1.bias'][index_conv]
        for i in range(128):
            w_attacker['conv2.weight'][i] = w_attacker['conv2.weight'][i][index_conv]

        w_attacker['fc2.weight'] = w_attacker['fc2.weight'][index_fc]
        w_attacker['fc2.bias'] = w_attacker['fc2.bias'][index_fc]
        for i in range(10):
            w_attacker['fc3.weight'][i] = w_attacker['fc3.weight'][i][index_fc]

        for key in w_attacker.keys():
            w_attacker[key] = torch.clamp(w_attacker[key], -0.99, 0.99)
        # test
        with torch.no_grad():
            test_model_honest = CNNCifar(args=args).to(args.device)
            test_model_honest.load_state_dict(w)
            test_accuracy_honest, client_loss_honest = test_img(test_model_honest, dataset_test, args)

            test_model_attacker = CNNCifar(args=args).to(args.device)
            test_model_attacker.load_state_dict(w_attacker)
            test_accuracy_attacker, client_loss_attacker = test_img(test_model_attacker, dataset_test, args)

        logging.info("before scale and exchange：")
        logging.info("honest accuracy : {}, loss : {}".format(test_accuracy_honest, client_loss_honest))
        logging.info("after scale and exchange：")
        logging.info("attacker accuracy : {}, loss : {}".format(test_accuracy_attacker, client_loss_attacker))
        w = copy.deepcopy(w_attacker)

    if args.attack_type == "gaussian_attack":
        w_to_np = []
        for key in w.keys():
            w_to_np += w[key].flatten().tolist()
        w_to_np = np.array(w_to_np)
        w_mean = np.mean(w_to_np)
        w_std = np.std(w_to_np)
        logging.info("w's mean : {}, w's std : {}".format(w_mean, w_std))
        w_gaussian = copy.deepcopy(w)
        for k in w.keys():
            noise = np.random.normal(w_mean, w_std, w[k].size())
            # if args.gpu != -1:
            #     noise = torch.from_numpy(noise).float().cuda()
            # else:
            noise = torch.from_numpy(noise).float()
            w_gaussian[k] = noise
        w = copy.deepcopy(w_gaussian)
    return w


def generate_chosen_attack_clients(args):
    if args.num_users <= 0 or args.comprised_rate <= 0:
        raise ValueError("num_users and comprised_rate must greater than 0")

    comprised_client = int(args.comprised_rate * args.num_users)
    if comprised_client > args.num_users:
        raise ValueError("comprised_rate setting resulted in the number of compromised_client exceeding num_users")

    chosen_attack_client = random.sample(range(args.num_users), comprised_client)
    chosen_attack_client.sort()
    return chosen_attack_client


def generate_scaling_factor(shuffle_ratio, round_idx):
    len_conv = int(shuffle_ratio * 64)
    len_fc = int(shuffle_ratio * 256)

    index_conv = random.sample(range(0, len_conv), len_conv) + list(range(len_conv, 64))
    index_fc = random.sample(range(0, len_fc), len_fc) + list(range(len_fc, 256))

    scale_conv = [random.uniform(1, 2) if (i + round_idx) % 2 == 0 else random.uniform(0.5, 1) for i in range(64)]
    scale_fc = [random.uniform(1, 2) if (i + round_idx) % 2 == 0 else random.uniform(0.5, 1) for i in range(256)]

    return index_conv, index_fc, scale_conv, scale_fc


def process_attackers(attacker_nums, w_locals):
    w_attackers = w_locals[:attacker_nums]
    noise_scale_attackers = copy.deepcopy(privacy_account(args, True,
                                                          num_items_train))

    for idx in range(attacker_nums):
        w_attackers[idx], l2norm_attackers = copy.deepcopy(clipping(args, w_attackers[idx]))

    w_attackers = noise_add(args, noise_scale_attackers, w_attackers)

    return w_attackers


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    now = int(time.time())
    timeArray = time.localtime(now)
    Time = time.strftime("%m.%d", timeArray)

    wandb.init(
        project="{}-{}-{}-{}".format(Time, str(args.aggregation_methods), str(args.dataset),
                                     str(args.model)),
        name=str(args.attack_type) + " comprised rate：" + str(args.comprised_rate) + "-local epoch：" + str(
            args.local_ep),
        config=args
    )

    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True

    # load dataset and split users
    if args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dict_users = cifar(dataset_train, args.num_users)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    w_glob = net_glob.state_dict()

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[0])
    num_items_train = len(local.ldr_train.dataset)
    noise_scale = copy.deepcopy(privacy_account(args, False, num_items_train))
    logging.info("noise_scale: {}".format(noise_scale))

    number_attack_round = []
    chosen_attack_client = []

    if args.attack_type != 'no_attack':
        chosen_attack_client = generate_chosen_attack_clients(args)
    logging.info("comprised clients = {}".format(chosen_attack_client))

    for round_idx in range(args.epochs):
        logging.info("################Communication round : {}".format(round_idx))
        if args.client_optimizer == 'sgd':
            if round_idx > 0:
                args.lr = args.lr * args.lr_decay
            logging.info("round {},learning rate : {}".format(round_idx, args.lr))

        w_locals = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        logging.info("client_indexes = " + str(idxs_users))

        attackers = [i for i in idxs_users if i in chosen_attack_client]
        attacker_nums = len(attackers)

        number_attack_round.append(attacker_nums)

        logging.info("number of attackers in each round = {}".format(number_attack_round))
        logging.info("current round = {}".format(attacker_nums))
        logging.info("chosen attack clients = {}".format(attackers))

        non_attackers = [i for i in idxs_users if i not in attackers]
        sorted_idxs_users = attackers + non_attackers

        index_conv, index_fc, scale_conv, scale_fc = generate_scaling_factor(args.shuffle_ratio, round_idx)

        for idx in sorted_idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), id=idx)
            if idx in attackers:
                w_attack = copy.deepcopy(w)
                w = copy.deepcopy(attack_method(w_attack, scale_conv, scale_fc, index_conv, index_fc))
            w_locals.append(copy.deepcopy(w))

        if args.use_dp == True:
            logging.info("---------------using dp----------------")
            w_attackers = []
            if attacker_nums > 0 and args.attack_type == "MSA":
                w_attackers = process_attackers(attacker_nums, w_locals)

            # Clipping
            for idx in range(attacker_nums, int(args.frac * args.num_users)):
                w_locals[idx], l2norm = copy.deepcopy(clipping(args, w_locals[idx]))

            w_locals = noise_add(args, noise_scale, w_locals)
            if args.attack_type == "MSA":
                for idx in range(attacker_nums):
                    w_locals[idx] = copy.deepcopy(w_attackers[idx])

        w_glob = aggregation(args, w_locals)

        net_glob.load_state_dict(w_glob)

        if round_idx == args.epochs - 1:
            train_acc, train_loss = test_img(net_glob, dataset_train, args)

            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx}, step=round_idx + 1)
            wandb.log({"Train/Loss": train_loss, "round": round_idx}, step=round_idx + 1)
            logging.info(stats)

            test_acc, test_loss = test_img(net_glob, dataset_test, args)
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx}, step=round_idx + 1)
            wandb.log({"Test/Loss": test_loss, "round": round_idx}, step=round_idx + 1)
            logging.info(stats)

        elif round_idx % args.frequency_of_the_test == 0:
            train_acc, train_loss = test_img(net_glob, dataset_train, args)
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx}, step=round_idx + 1)
            wandb.log({"Train/Loss": train_loss, "round": round_idx}, step=round_idx + 1)
            logging.info(stats)

            test_acc, test_loss = test_img(net_glob, dataset_test, args)
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx}, step=round_idx + 1)
            wandb.log({"Test/Loss": test_loss, "round": round_idx}, step=round_idx + 1)
            logging.info(stats)
