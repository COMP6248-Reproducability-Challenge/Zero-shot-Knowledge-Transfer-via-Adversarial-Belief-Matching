mode = "zero_shot"
dataset = "cifar10"
seed = 0
save_path = "../PreTrainedModels"

teacher = dict(
    depth = 40,
    widen_factor = 2,
    dropRate = 0.0,
    input_features = 3,
    output_features = 16,
    strides = [1, 2, 2]
)

student = dict(
    depth = 16,
    widen_factor = 1,
    dropRate = 0.0,
    input_features = 3,
    output_features = 16,
    strides = [1, 2, 2]
)

generator = dict(
    input_dim= 100
)

downsample = dict(
    action=False,
    value= 200
)