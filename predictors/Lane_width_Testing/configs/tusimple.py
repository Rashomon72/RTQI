# DATA
dataset='Tusimple'
data_root = 'C:/Users/Faculty/Downloads/test-images'

# TRAIN
epoch = 40
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = "C:/Users/Faculty/Downloads/test-images/logs"

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None
annotation_file_name = "test_annotations"
output_dir = "output_images"

num_lanes = 4
