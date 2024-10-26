import os
import os.path as osp
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm
import argparse

def main(input_dir, output_dir):
    for i in range(11):
        seq_id = '{:02d}'.format(i)
        file_names = glob.glob(osp.join(input_dir, seq_id, 'velodyne', '*.bin'))
        for file_name in tqdm(file_names):
            frame = file_name.split('/')[-1][:-4]
            new_file_name = osp.join(output_dir, seq_id, frame + '.npy')
            
            os.makedirs(osp.dirname(new_file_name), exist_ok=True)

            points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
            points = points[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd = pcd.voxel_down_sample(0.3)
            points = np.array(pcd.points).astype(np.float32)
            np.save(new_file_name, points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downsample LiDAR point clouds")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input sequences directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the output downsampled directory")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
