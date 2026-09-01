import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T
import random


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split,
        train_ratio=0.625,  # Default: 5/8 cameras for training
        random_seed=42
    ):
        """
        Modified version with train/test camera splitting
        
        Args:
            cam_extrinsics: Dictionary of camera extrinsics
            cam_intrinsics: Dictionary of camera intrinsics  
            cam_folder: Path to camera folders
            split: "train" or "test"
            train_ratio: Ratio of cameras to use for training (default 5/8)
            random_seed: Random seed for reproducible splitting
        """
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        
        # Split cameras into train/test sets
        self.train_cam_keys, self.test_cam_keys = self.split_cameras(
            cam_extrinsics, train_ratio, random_seed
        )
        
        print(f"Train cameras: {len(self.train_cam_keys)}")
        print(f"Test cameras: {len(self.test_cam_keys)}")
        print(f"Split: {split}")
        
        self.image_paths, self.image_poses, self.image_times = self.load_images_path(
            cam_folder, cam_extrinsics, cam_intrinsics, split
        )
        
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)
            print("test split :",len(self.image_paths))
    
    def split_cameras(self, cam_extrinsics, train_ratio, random_seed):
        """Split cameras into train and test sets"""
        all_cam_keys = list(cam_extrinsics.keys())
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        random.shuffle(all_cam_keys)
        
        # Calculate split indices
        num_train = max(1, int(len(all_cam_keys) * train_ratio))
        num_test = len(all_cam_keys) - num_train
        
        train_cam_keys = all_cam_keys[:num_train]
        test_cam_keys = all_cam_keys[num_train:]
        
        return train_cam_keys, test_cam_keys
    
    def load_images_path(self, cam_folder, cam_extrinsics, cam_intrinsics, split):
        """Load image paths for the specified split"""
        # Check if cam01 exists, if not find first existing camera folder
        cam_folders = [f for f in os.listdir(cam_folder) if f.startswith('cam')]
        if not cam_folders:
            raise ValueError(f"No camera folders found in {cam_folder}")
        
        # Use first camera folder to determine image length
        first_cam_folder = sorted(cam_folders)[0]
        image_length = len(os.listdir(os.path.join(cam_folder, first_cam_folder)))
        self.image_length = image_length
        print("Total image length per camera:", image_length)
        
        # Determine which cameras to use based on split
        if split == "train":
            cam_keys_to_use = self.train_cam_keys
        elif split == "test":
            cam_keys_to_use = self.test_cam_keys
        else:
            raise ValueError(f"Invalid split: {split}")
        
        image_paths = []
        image_poses = []
        image_times = []
        
        print(f"Loading images from {len(cam_keys_to_use)} cameras for {split} split")
        
        for key in cam_keys_to_use:
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            
            # Extract camera number from name
            cam_name = os.path.basename(extr.name)
            if 'cam' in cam_name.lower():
                print('-'*50)
                print("cam name:", cam_name)
                print('-'*50)
                # Try to extract camera number (handles formats like "cam01", "camera1", etc.)
                number = None
                for i, char in enumerate(cam_name):
                    if char.isdigit():
                        # Found first digit, extract the number
                        digits = []
                        for j in range(i, len(cam_name)):
                            if cam_name[j].isdigit():
                                digits.append(cam_name[j])
                            else:
                                break
                        number = ''.join(digits)
                        number = str(int(number)+1)
                        print("cam index:", number)
                        break
                
                if number is None:
                    # Fallback: use the camera key index
                    number = str(int(''.join(filter(str.isdigit, key))) if any(char.isdigit() for char in key) else "1")
            else:
                number = "1"
            
            images_folder = os.path.join(cam_folder, f"cam{number.zfill(2)}")
            print(f"Loading from: {images_folder}")
            
            # Determine which frames to load
            if split == "train":
                # For training, load most frames (e.g., all but leave some for validation)
                if image_length <= 3:
                    image_range = range(image_length)
                else:
                    # Load majority of frames for training
                    image_range = range(image_length - 1)  # Leave last frame for validation
            else:  # test split
                # For testing, load fewer frames to evaluate novel view synthesis
                if image_length == 1 or image_length == 2:
                    image_range = [image_range[0]]
                else:
                    # Load representative frames for evaluation
                    image_range = [0, image_length//2, image_length-1]
            
            for i in image_range:    
                num = i + 1
                image_path = os.path.join(
                    images_folder, 
                    f"cam_{number.zfill(4)}_{str(num).zfill(4)}.jpg"
                )
                
                # Check if file exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image path does not exist: {image_path}")
                    continue
                
                image_paths.append(image_path)
                image_poses.append((R, T))
                image_times.append(float(i/image_length))
        
        print(f"Loaded {len(image_paths)} images for {split} split")
        return image_paths, image_poses, image_times
    
    def get_video_cam_infos(self, datadir):
        """Generate novel camera poses for video rendering"""
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        
        # Use first image to get dimensions
        if self.image_paths:
            image = Image.open(self.image_paths[0])
            image = self.transform(image)
        else:
            # Fallback: use default dimensions
            image = Image.new('RGB', (self.FovX, self.FovY))
            image = self.transform(image)

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time))
        return cameras
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    
    def load_pose(self, index):
        return self.image_poses[index]
    
    def get_camera_splits(self):
        """Return the camera splits for inspection"""
        return {
            'train_cameras': self.train_cam_keys,
            'test_cameras': self.test_cam_keys
        }
