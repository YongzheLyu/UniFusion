import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split,
        train_cameras=None,
        test_cameras=None,
        subset_start=None,
        subset_end=None
    ):
        self.subset_start = subset_start
        self.subset_end = subset_end
        
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        
            
        self.transform = T.ToTensor()
        self.split = split
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)
            #print(len(self.video_cam_infos))
            print("test split :",len(self.image_paths))
        if subset_start is not None and subset_end is not None:
            self.image_paths = self.image_paths[subset_start:subset_end]
            self.image_poses = self.image_poses[subset_start:subset_end]
            self.image_times = self.image_times[subset_start:subset_end]
            
        
    
    def extract_camera_number(self, extr_name):
        """Extract camera number from extrinsic file name in a more robust way."""
        base_name = os.path.basename(extr_name)
        # Try different patterns to extract camera number
        import re
        matches = re.findall(r'cam0*(\d+)', base_name, re.IGNORECASE)
        if matches:
            return int(matches[0])
        # Fallback: try standard position-based parsing
        if len(base_name) > 7 and base_name[7].isdigit():
            return int(base_name[7])
        elif len(base_name) > 4 and base_name[4].isdigit():
            return int(base_name[4]) + 1
        else:
            print(f"[WARNING] Cannot extract camera number from {base_name}, returning 1")
            return 1

    def get_camera_split_mapping(self, cam_extrinsics, split):
        """Determine which cameras should be used for training/testing split."""
        # Extract camera numbers from all available cameras
        camera_numbers = {}
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            cam_num = self.extract_camera_number(extr.name)
            camera_numbers[idx] = cam_num

        # Sort cameras by number
        sorted_cameras = sorted(camera_numbers.items(), key=lambda x: x[1])
        all_camera_idxs = [item[0] for item in sorted_cameras]

        print(f"[INFO] Found {len(all_camera_idxs)} total cameras: {[camera_numbers[idx] for idx in all_camera_idxs]}")

        # Split: 5 training cameras, 3 testing cameras
        n_train = 5
        n_test = 3

        if len(all_camera_idxs) < n_train + n_test:
            print(f"[WARNING] Only {len(all_camera_idxs)} cameras available, need {n_train + n_test}")
            # Use first 5 for training if insufficient cameras
            if split == "train":
                return all_camera_idxs[:min(len(all_camera_idxs), n_train+1)]
            else:
                return all_camera_idxs[n_train:] if len(all_camera_idxs) > n_train else []

        # Default split: first 5 for training, next 3 for testing
        if split == "train":
            return all_camera_idxs[:n_train]
        else:  # test
            return all_camera_idxs[n_train:n_train + n_test]

    def load_images_path(self, cam_folder, cam_extrinsics, cam_intrinsics, split):
        # Load total image length from cam01 folder or any existing cam folder
        total_cameras = sum(1 for key in cam_extrinsics if os.path.basename(cam_extrinsics[key].name).startswith('cam'))
        print(f"[INFO] Total cameras found: {total_cameras}")

        # Determine camera listing
        try:
            # Try to find an existing camera folder to determine image length
            cam_dirs = [d for d in os.listdir(cam_folder) if d.startswith('cam') and os.path.isdir(os.path.join(cam_folder, d))]
            if cam_dirs:
                existing_cam = sorted(cam_dirs)[0]
                image_length = len([f for f in os.listdir(os.path.join(cam_folder, existing_cam)) if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))])
            else:
                print("[WARNING] No camera folders found, trying default cam01 format")
                image_length = len(os.listdir(os.path.join(cam_folder, "cam01")))
        except:
            print("[WARNING] Could not determine image length, using default 50")
            image_length = 50

        self.image_length = image_length
        print(f"[INFO] Total image length per camera: {image_length}")

        # Get camera split mapping - this determines which cameras to use
        valid_camera_idxs = self.get_camera_split_mapping(cam_extrinsics, split)

        image_paths = []
        image_poses = []
        image_times = []

        print(f"[INFO] Loading {len(valid_camera_idxs)} cameras for {split} split")

        for idx in valid_camera_idxs:
            key = list(cam_extrinsics.keys())[idx]
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            camera_num = self.extract_camera_number(extr.name)+1
            images_folder = os.path.join(cam_folder, f"cam{camera_num:02d}")

            if not os.path.exists(images_folder):
                print(f"[WARNING] Camera folder not found: {images_folder}, skipping camera {camera_num}")
                continue

            print(f"[INFO] Loading {split} images from camera {camera_num}: {images_folder}")

            # Determine which frames to load
            image_range = range(image_length)
            if split == "test" and image_length > 2:
                # For test, use 1/5 of the frames scattered through the sequence (every 5th frame)
                step = 5
                image_range = [i for i in range(0, image_length, step)]

            for i in image_range:
                num = i + 1
                image_name = f"cam_{camera_num:04d}_{num:04d}.jpg"
                image_path = os.path.join(images_folder, image_name)

                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    image_poses.append((R, T))
                    image_times.append(float(i/image_length))
                else:
                    print(f"[WARNING] Image not found: {image_path}")

        print(f"[INFO] Successfully loaded {len(image_paths)} images for {split} split")
        return image_paths, image_poses, image_times
    #bug
    def get_video_cam_infos(self,datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        #print(self.image_path[0])
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        for idx, p in enumerate(val_poses):
            #print("video camera idx:", idx)
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
        #print(len(self.image_paths))
        return len(self.image_paths)
    def __getitem__(self, index):
        #print(index)
        #print(len(self.image_paths))
        #print(self.image_paths)
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]