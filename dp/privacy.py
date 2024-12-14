import numpy as np

def privacy_account(args, is_attack, num_items_train):
    q_s = args.frac
    delta_s = 2*args.clipthr/num_items_train

    if is_attack:
        privacy_budget = args.attackers_privacy_budget
    else:
        privacy_budget = args.privacy_budget

    noise_scale = delta_s * np.sqrt(2 * q_s * args.epochs * np.log(1 / args.delta)) / privacy_budget

    return noise_scale


