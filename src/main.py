import No_Teacher
import zero_shot
import KD_AT
import argparse
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZEROSHOT')
    parser.add_argument("--m", choices=["no_teacher", "zero_shot", "kd_at"], required=True, type=str, help="Method")
    parser.add_argument("--d", choices=["cifar10", "svhn"], required=True, type=str, help="Dataset")
    parser.add_argument("--r", action='store_true' , help="Reproducibility Mode: Setups seeds")
    parser.add_argument("--s", required=False, type=str, default="../PretrainedModels", help="Saving path")

    args = parser.parse_args()

    args.m = args.m.lower()
    args.d = args.d.lower()

    seed = 0
    if args.r:
        utils.setup_seeds(seed)

    if args.m == "no_teacher":
        No_Teacher.No_teacher(args.s, args.d, seed)
    elif args.m == "zero_shot":
        zeros = zero_shot.ZeroShot(100, args.d)
        zeros.train_ZS()
    else:
        kd_at = KD_AT.FewShotKT(100, args.d)
        kd_at.train_KT_AT()




