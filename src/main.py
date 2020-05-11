import No_Teacher
import zero_shot
import KD_AT
import argparse
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZEROSHOT')
    parser.add_argument("--m", choices=["no_teacher", "zero_shot", "kd_at"], required=True, type=str, help="Method")
    parser.add_argument("--d", choices=["cifar10", "svhn"], required=True, type=str, help="Dataset")
    parser.add_argument("--r", choices=[True,False], required=False, default=False, type=bool, help="Reproducibility mode: Setups seeds")

    args = parser.parse_args()

    args.m = args.m.lower()
    args.d = args.d.lower()

    if args.r:
        utils.setup_seeds()

    if args.m == "no_teacher":
        No_Teacher.No_teacher(args.d)
    elif args.m == "zero_shot":
        zeros = zero_shot.ZeroShot(100, args.d)
        zeros.train_ZS()
    else:
        kd_at = KD_AT.FewShotKT(100, args.d)
        kd_at.train_KT_AT()




