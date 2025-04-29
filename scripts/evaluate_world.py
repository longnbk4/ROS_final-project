#!/usr/bin/env python
import argparse
import os
import numpy as np
from lxml import etree
from scipy.ndimage import label

def evaluate_world_map(world_file, resolution=0.05, real_world_dims=None, real_world_area=None, real_free_area=None):
    # Đọc file .world
    try:
        tree = etree.parse(world_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Không thể đọc file {world_file}: {e}")
        return None

    # Giả định kích thước môi trường (có thể cần điều chỉnh theo file .world)
    map_width_m = 100.0  # Chiều rộng mặc định (m)
    map_height_m = 100.0  # Chiều cao mặc định (m)
    if real_world_dims:
        map_width_m, map_height_m = real_world_dims

    # Tạo lưới 2D
    width = int(map_width_m / resolution)
    height = int(map_height_m / resolution)
    map_grid = np.ones((height, width), dtype=np.uint8) * 205  # 205: Vùng tự do (như .pgm)

    # Phân tích các model trong file .world
    for model in root.findall(".//model"):
        # Lấy pose (vị trí) và geometry (kích thước)
        pose = model.find("pose")
        if pose is not None:
            x, y, z = map(float, pose.text.split()[:3])
            # Chỉ xử lý các model trên mặt phẳng XY (bỏ qua Z)
            if abs(z) > 0.1:  # Bỏ qua các model không nằm trên mặt đất
                continue

        # Lấy kích thước hình học (giả sử là box)
        geometry = model.find(".//collision/geometry/box/size")
        if geometry is not None:
            size_x, size_y, size_z = map(float, geometry.text.split())
            # Chuyển kích thước thành pixel
            size_x_pixels = int(size_x / resolution)
            size_y_pixels = int(size_y / resolution)

            # Chuyển vị trí thành chỉ số lưới
            x_pixel = int((x + map_width_m / 2) / resolution)
            y_pixel = int((y + map_height_m / 2) / resolution)

            # Đánh dấu vùng vật cản (0: Obstacle)
            x_start = max(0, x_pixel - size_x_pixels // 2)
            x_end = min(width, x_pixel + size_x_pixels // 2)
            y_start = max(0, y_pixel - size_y_pixels // 2)
            y_end = min(height, y_pixel + size_y_pixels // 2)
            map_grid[y_start:y_end, x_start:x_end] = 0

    # Tính toán các thông số
    total_pixels = width * height
    free_pixels = np.sum(map_grid == 205)
    obstacle_pixels = np.sum(map_grid == 0)
    free_area = free_pixels * (resolution ** 2)
    obstacle_area = obstacle_pixels * (resolution ** 2)
    total_mapped_area = free_area + obstacle_area
    total_area = total_pixels * (resolution ** 2)
    coverage_ratio = (total_mapped_area / total_area) * 100 if total_area > 0 else 0
    obstacle_ratio = (obstacle_area / total_mapped_area) * 100 if total_mapped_area > 0 else 0

    # Phân tích vùng vật cản và tự do
    obstacle_binary = (map_grid == 0).astype(np.uint8)
    labeled_obstacles, num_obstacles = label(obstacle_binary)
    free_binary = (map_grid == 205).astype(np.uint8)
    labeled_free, num_free_regions = label(free_binary)
    avg_free_region_size = np.mean([np.sum(labeled_free == i) * (resolution ** 2) for i in range(1, num_free_regions + 1)]) if num_free_regions > 0 else 0

    # Tính sai số
    width_error = abs(map_width_m - real_world_dims[0]) / real_world_dims[0] * 100 if real_world_dims else None
    height_error = abs(map_height_m - real_world_dims[1]) / real_world_dims[1] * 100 if real_world_dims else None
    area_error = abs(total_area - real_world_area) / real_world_area * 100 if real_world_area else None
    free_area_error = abs(free_area - real_free_area) / real_free_area * 100 if real_free_area else None

    # Kết quả
    result = {
        "Map Name": os.path.basename(world_file).replace('.world', ''),
        "Chiều rộng (m)": map_width_m,
        "Chiều dài (m)": map_height_m,
        "Diện tích tổng (m²)": total_area,
        "Diện tích đã quét (m²)": total_mapped_area,
        "Diện tích vùng tự do (m²)": free_area,
        "Diện tích vùng vật cản (m²)": obstacle_area,
        "Khả năng quét (%)": coverage_ratio,
        "Tỷ lệ vật cản (%)": obstacle_ratio,
        "Số vùng vật cản riêng biệt": num_obstacles,
        "Số vùng tự do riêng biệt": num_free_regions,
        "Diện tích trung bình vùng tự do (m²)": avg_free_region_size,
        "Sai số chiều rộng (%)": width_error,
        "Sai số chiều dài (%)": height_error,
        "Sai số diện tích tổng (%)": area_error,
        "Sai số diện tích vùng tự do (%)": free_area_error
    }

    print(f"\n=== Đánh giá bản đồ: {result['Map Name']} ===")
    for key, value in result.items():
        if value is None:
            continue
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    return result

def evaluate_map(map_name, map_dir, real_world_dims=None, real_world_area=None, real_free_area=None):
    world_file = os.path.join(map_dir, f"{map_name}.world")
    if not os.path.exists(world_file):
        print(f"Không tìm thấy file {world_file}")
        return
    evaluate_world_map(world_file, real_world_dims=real_world_dims, real_world_area=real_world_area, real_free_area=real_free_area)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá bản đồ từ file .world.")
    parser.add_argument("map_name", type=str, help="Tên bản đồ cần đánh giá (ví dụ: maze)")
    args = parser.parse_args()

    map_dir = os.path.expanduser("~/catkin_ws/src/xerobotvisai2/worlds/")
    real_world_dims = (100.0, 100.0)
    real_world_area = 10000.0
    real_free_area = 9795.0

    evaluate_map(args.map_name, map_dir, real_world_dims, real_world_area, real_free_area)