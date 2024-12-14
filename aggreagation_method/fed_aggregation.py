import copy
import logging
from collections import OrderedDict, defaultdict

import functools
import torch
import numpy as np

def row_into_parameters(row, grad):
    offset = 0
    for name in grad.keys():
        new_size = functools.reduce(lambda x, y: x * y, grad[name].shape)
        current_data = row[offset:offset + new_size]

        grad[name][:] = torch.from_numpy(current_data.reshape(grad[name].shape))
        offset += new_size
    return grad

def _krum_create_distances(users_grads):
    distances = defaultdict(dict)
    for i in range(len(users_grads)):
        for j in range(i):
            distances[i][j] = distances[j][i] = np.linalg.norm(users_grads[i] - users_grads[j], ord=1)
    return distances


def krum(users_grads, users_count, corrupted_count, distances=None, return_index=False):
    if not return_index:
        assert users_count >= 2 * corrupted_count + 1, (
            'users_count>=2*corrupted_count + 3', users_count, corrupted_count)
    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1

    if distances is None:
        distances = _krum_create_distances(users_grads)
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        logging.info("user {}:current_error:{}".format(user, current_error))
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        logging.info("selected index：{},minimal_error:{}".format(minimal_error_index, minimal_error))
        return users_grads[minimal_error_index]


def aggregation(args, w):

    if args.aggregation_methods == 'fedavg':
        w_avg = copy.deepcopy(w[0])
        if isinstance(w[0], np.ndarray) == True:
            for i in range(1, len(w)):
                w_avg += w[i]
            w_avg = w_avg / len(w)
        else:
            for k in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[k] += w[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg
    elif args.aggregation_methods == 'krum':
        w_locals = []
        for i in range(len(w)):
            w_local = []
            for key in w[i].keys():
                w_local += w[i][key].flatten().tolist()
            w_locals.append(w_local)
        w_locals = np.array(w_locals)

        exclude_ratio = 0.2
        client_num_per_round = int(args.num_users * args.frac)
        w_glob = krum(w_locals, client_num_per_round, int(client_num_per_round * exclude_ratio))
        model = row_into_parameters(w_glob, w[0])
        aggre_model = copy.deepcopy(model)
        return aggre_model
    elif args.aggregation_methods == 'trimmed_mean':
        remove = int(0.1 * len(w))

        temp_updates = []
        for k in w[0].keys():
            stacked = torch.stack([i[k] for i in w])
            sorted_stack, _ = torch.sort(stacked, dim=0)
            sorted_stack = sorted_stack[
                           remove: len(w) - remove]
            temp_updates.append((k, [torch.flatten(i, start_dim=0, end_dim=1) for i in
                                     list(torch.split(sorted_stack, split_size_or_sections=1, dim=0))]))

        w_locals = []
        for i in range(len(w) - 2 * remove):
            curr = OrderedDict()
            for j in temp_updates:
                curr[j[0]] = j[1][i]
            w_locals.append(curr)

        w_avg = copy.deepcopy(w_locals[0])
        if isinstance(w_locals[0], np.ndarray) == True:
            for i in range(1, len(w_locals)):
                w_avg += w_locals[i]
            w_avg = w_avg / len(w_locals)
        else:
            for k in w_avg.keys():
                for i in range(1, len(w_locals)):
                    w_avg[k] += w_locals[i][k]
                w_avg[k] = torch.div(w_avg[k], len(w_locals))
        return w_avg


