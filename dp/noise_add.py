import numpy as np
import copy
import torch


def noise_add(args, noise_scale, w):
    w_noise = copy.deepcopy(w)
    # w_noise = np.array(w_noise).astype(float)
    # w_noise = torch.from_numpy(w_noise)
    if isinstance(w[0], np.ndarray) == True:
        noise = np.random.normal(0, noise_scale, w.size())
        w_noise = w_noise + noise

    for k in range(len(w)):
        for i in w[k].keys():
            noise = np.random.normal(0, noise_scale, w[k][i].size())
            # if args.gpu != -1:
            #     noise = torch.from_numpy(noise).float().cuda()
            # else:
            noise = torch.from_numpy(noise).float().cuda()
            w_noise[k][i] = w_noise[k][i].cuda() + noise
    # elif args.aggregation_methods == "trimmed_mean" and len(w_noise) != 10:
    #     for k in range(len(w)):
    #         for i in w[k].keys():
    #            noise = np.random.normal(0,noise_scale,w[k][i].size())
    #            noise = torch.from_numpy(noise).float()
    #            noise = -np.sign(w[k][i].cpu()) * abs(noise)
    #            noise = noise.numpy()
    #            # if args.gpu != -1:
    #            #     noise = torch.from_numpy(noise).float().cuda()
    #            # else:
    #            noise = torch.from_numpy(noise).float().cuda()
    #            w_noise[k][i] = w_noise[k][i] + noise
    return w_noise


def users_sampling(args, w, chosenUsers):
    if args.num_chosenUsers < args.num_users:
        w_locals = []
        for i in range(len(chosenUsers)):
            w_locals.append(w[chosenUsers[i]])
    else:
        w_locals = copy.deepcopy(w)
    return w_locals


def clipping(args, w):
    if get_norm(w) > args.clipthr:
        w_local = copy.deepcopy(w)
        for i in w.keys():
            w_local[i] = copy.deepcopy(w[i] * args.clipthr / get_norm(w))
    else:
        w_local = copy.deepcopy(w)
    return w_local, get_norm(w)


def get_norm(params_a):
    sum = 0
    if isinstance(params_a, np.ndarray):
        sum += pow(np.linalg.norm(params_a, ord=2), 2)
    else:
        for i in params_a.keys():
            if len(params_a[i]) == 1:
                sum += pow(np.linalg.norm(params_a[i].cpu().numpy(), ord=2), 2)
            else:
                a = copy.deepcopy(params_a[i].cpu().numpy())
                for j in a:
                    x = copy.deepcopy(j.flatten())
                    sum += pow(np.linalg.norm(x, ord=2), 2)
    norm = np.sqrt(sum)
    return norm
