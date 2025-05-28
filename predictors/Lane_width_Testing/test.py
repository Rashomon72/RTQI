import json
import matplotlib.pyplot as plt
import matplotlib
import cv2
from predictors.Lane_width_Testing.evaluation.eval_wrapper import eval_lane
from predictors.Lane_width_Testing.utils.common import merge_config
from predictors.Lane_width_Testing.model.model import parsingNet
import torch
import os
from statistics import mode

# Limit PyTorch to use only a single thread for CPU operations
torch.set_num_threads(1)

# You can also set the number of inter-op parallelism threads to 1
torch.set_num_interop_threads(1)

matplotlib.use('Agg')

all_width = []


def get_lane_points(lane, h_samples):
    return [(x, y) for x, y in zip(lane, h_samples) if x != -2]


def annotate_and_save(image_path, json_data, output_dir, serial_number):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lanes = json_data['lanes']
    h_samples = json_data['h_samples']

    left_lane_points = get_lane_points(lanes[1], h_samples)
    right_lane_points = get_lane_points(lanes[2], h_samples)

    if len(left_lane_points) >= 5 and len(right_lane_points) >= 5:
        width = abs(left_lane_points[-5][0] - right_lane_points[-5][0])

        plt.figure(figsize=(10, 5))
        plt.imshow(image_rgb)
        plt.plot([p[0] for p in left_lane_points], [p[1]
                 for p in left_lane_points], 'go-')
        plt.plot([p[0] for p in right_lane_points], [p[1]
                 for p in right_lane_points], 'go-')
        plt.plot([left_lane_points[-5][0], right_lane_points[-5][0]],
                 [left_lane_points[-5][1], right_lane_points[-5][1]], 'r-', linewidth=2)
        plt.text((left_lane_points[-5][0] + right_lane_points[-5][0]) / 2, left_lane_points[-5][1] - 10,
                 f'Width: {width}px', color='red', fontsize=12, ha='center')

        filename = f"annotated_image_{serial_number}.jpg"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()
        all_width.append(width)
        # print("(●'◡'●) Width:", width)
    else:
        all_width.append(0)


def predict_lane_width(data_root, annotation_file_name, test_model, output_dir, images_folder):
    class DefaultConfig:
        def __init__(self, data_root, annotation_file_name, test_model, output_dir):
            self.data_root = data_root
            self.annotation_file_name = annotation_file_name
            self.test_model = test_model
            self.output_dir = output_dir
            self.test_work_dir = data_root
            self.backbone = "18"
            self.dataset = "Tusimple"
            self.griding_num = 100
            self.num_lanes = 4

    # Create an instance of DefaultConfig with dynamic values
    cfg = DefaultConfig(data_root, annotation_file_name,
                        test_model, output_dir)

    # Folder where images are present for testing
    folder_name = images_folder
    folder_path = os.path.join(cfg.data_root, folder_name).replace("\\", "/")
    # File used for testing, has all the relative paths (data_root) of the images
    output_file = os.path.join(cfg.data_root, "test.txt").replace("\\", "/")

    with open(output_file, 'w') as f:
        for image_name in os.listdir(folder_path):
            if image_name.endswith(('.jpg', '.jpeg', '.png')):
                f.write(f"{folder_name}/{image_name}\n")

    print('Start testing...')
    cls_num_per_lane = 56 if cfg.dataset == 'Tusimple' else 18

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes),
                     use_aux=False).to(device)

    state_dict = cfg.test_model
    net.load_state_dict(
        {k[7:] if 'module.' in k else k: v for k, v in state_dict.items()}, strict=False)

    os.makedirs(cfg.test_work_dir, exist_ok=True)
    eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir,
              cfg.griding_num, False, cfg.annotation_file_name)

    ann_path = os.path.join(
        cfg.data_root, f"{cfg.annotation_file_name}.txt").replace("\\", "/")
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(ann_path, 'r') as f:
        for serial_number, line in enumerate(f, start=1):
            json_data = json.loads(line)
            image_path = os.path.join(cfg.data_root, json_data['raw_file'])
            annotate_and_save(image_path, json_data,
                              cfg.output_dir, serial_number)

    print("Annotation and saving complete.")

    # mean = sum(all_width) / len(all_width) if all_width else 0
    res_mode = mode(all_width) if all_width else 0

    return res_mode
