#!/usr/bin/env python
import cv2
import numpy as np
import yaml
import os
import argparse
from scipy.ndimage import label

def evaluate_single_map(pgm_file, yaml_file, real_world_dims=None, real_world_area=None, real_free_area=None):
    with open(yaml_file, 'r') as f:
        map_config = yaml.safe_load(f)
    
    resolution = map_config['resolution']
    origin = map_config['origin']

    real_width = map_config.get('real_width', None)
    real_height = map_config.get('real_height', None)
    real_area = map_config.get('real_area', None)
    real_free_area_in_yaml = map_config.get('real_free_area', None)

    if real_width and real_height:
        real_world_dims = (real_width, real_height)
    if real_area:
        real_world_area = real_area
    if real_free_area_in_yaml:
        real_free_area = real_free_area_in_yaml

    map_img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        print(f"Không thể đọc file {pgm_file}")
        return None

    height, width = map_img.shape

    map_width_m = width * resolution
    map_height_m = height * resolution

    total_pixels = width * height

    unknown_pixels = np.sum(map_img == 255)
    free_pixels = np.sum(map_img == 205)
    obstacle_pixels = np.sum(map_img == 0)

    free_area = free_pixels * (resolution ** 2)
    obstacle_area = obstacle_pixels * (resolution ** 2)
    total_mapped_area = free_area + obstacle_area
    total_area = total_pixels * (resolution ** 2)

    coverage_ratio = (total_mapped_area / total_area) * 100 if total_area > 0 else 0
    obstacle_ratio = (obstacle_area / total_mapped_area) * 100 if total_mapped_area > 0 else 0

    obstacle_binary = (map_img == 0).astype(np.uint8)
    labeled_obstacles, num_obstacles = label(obstacle_binary)

    free_binary = (map_img == 205).astype(np.uint8)
    labeled_free, num_free_regions = label(free_binary)
    if num_free_regions > 0:
        free_region_sizes = [np.sum(labeled_free == i) * (resolution ** 2) for i in range(1, num_free_regions + 1)]
        avg_free_region_size = np.mean(free_region_sizes)
    else:
        avg_free_region_size = 0

    width_error = None
    height_error = None
    area_error = None
    free_area_error = None
    if real_world_dims:
        real_width, real_height = real_world_dims
        width_error = abs(map_width_m - real_width) / real_width * 100 if real_width > 0 else 0
        height_error = abs(map_height_m - real_height) / real_height * 100 if real_height > 0 else 0
    if real_world_area:
        area_error = abs(total_area - real_world_area) / real_world_area * 100 if real_world_area > 0 else 0
    if real_free_area:
        free_area_error = abs(free_area - real_free_area) / real_free_area * 100 if real_free_area > 0 else 0

    result = {
        "Map Name": os.path.basename(pgm_file).replace('.pgm', ''),
        "Chiều rộng": map_width_m,
        "Chiều dài": map_height_m,
        "Diện tích tổng": total_area,
        "Diện tích đã quét": total_mapped_area,
        "Diện tích vùng tự do": free_area,
        "Diện tích vùng vật cản": obstacle_area,
        "Khả năng quét": coverage_ratio,
        "Tỷ lệ vật cản": obstacle_ratio,
        "Số vùng vật cản riêng biệt": num_obstacles,
        "Số vùng tự do riêng biệt": num_free_regions,
        "Diện tích trung bình vùng tự do": avg_free_region_size,
        "Sai số chiều rộng": width_error,
        "Sai số chiều dài": height_error,
        "Sai số diện tích tổng": area_error,
        "Sai số diện tích vùng tự do": free_area_error
    }

    print(f"\n=== Đánh giá bản đồ: {result['Map Name']} ===")
    for key, value in result.items():
        if value is None:
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    return result

def evaluate_map(map_name, map_dir, real_world_dims=None, real_world_area=None, real_free_area=None):
    pgm_file = os.path.join(map_dir, f"{map_name}.pgm")
    yaml_file = os.path.join(map_dir, f"{map_name}.yaml")

    if not os.path.exists(pgm_file):
        print(f"Không tìm thấy file {pgm_file}")
        return
    if not os.path.exists(yaml_file):
        print(f"Không tìm thấy file {yaml_file}")
        return

    evaluate_single_map(pgm_file, yaml_file, real_world_dims, real_world_area, real_free_area)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá một bản đồ cụ thể từ terminal.")
    parser.add_argument("map_name", type=str, help="Tên bản đồ cần đánh giá (ví dụ: mazemap1)")
    parser.add_argument("--width", type=float, default=10.0, help="Chiều rộng thực tế (m)")
    parser.add_argument("--height", type=float, default=10.0, help="Chiều dài thực tế (m)")
    parser.add_argument("--area", type=float, default=100.0, help="Diện tích tổng thực tế (m²)")
    parser.add_argument("--free_area", type=float, default=97.95, help="Diện tích vùng tự do thực tế (m²)")
    args = parser.parse_args()

    map_dir = os.path.expanduser("~/catkin_ws/src/xerobotvisai2/maps/")

    real_world_dims = (args.width, args.height)
    real_world_area = args.area
    real_free_area = args.free_area

    evaluate_map(args.map_name, map_dir, real_world_dims, real_world_area, real_free_area)