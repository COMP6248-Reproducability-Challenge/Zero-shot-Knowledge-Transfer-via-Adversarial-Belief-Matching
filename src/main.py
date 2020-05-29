import os
import pathlib

import KD_AT
import No_Teacher
import belief_match
import config
import utils
import zero_shot

if __name__ == "__main__":
    utils.setup_seeds(config.seed)

    os.chdir(pathlib.Path(__file__).parent.absolute())

    if config.goal == "belief_match":
        belief = belief_match.BeliefMatch()
        belief.calculate()
    else:
        mode = config.mode

        if mode == "no_teacher":
            no_teacher = No_Teacher.No_teacher()
            if not config.test_mode:
                no_teacher.train()
            no_teacher.test()
        elif mode == "zero_shot":
            zeros = zero_shot.ZeroShot()
            if not config.test_mode:
                zeros.train()
            zeros.test(test=True)
        elif mode == "kd_at":
            kd_at = KD_AT.FewShotKT()
            if not config.test_mode:
                kd_at.train_KT_AT()
            kd_at.test(test=True)
        else:
            raise ValueError('Not valid mode')
