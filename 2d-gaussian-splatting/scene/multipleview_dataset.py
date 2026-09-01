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
        split
    ):
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)
            #print(len(self.video_cam_infos))
            print("test split :",len(self.image_paths))
        
    
    def load_images_path(self, cam_folder, cam_extrinsics,cam_intrinsics,split):
        image_length = len(os.listdir(os.path.join(cam_folder,"cam01")))
        self.image_length=image_length
        print("Total image length per camera:",image_length)
        #len_cam=len(cam_extrinsics)
        image_paths=[]
        image_poses=[]
        image_times=[]
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            #print(os.path.basename(extr.name))
            #print(os.path.basename(extr.name))
            number = os.path.basename(extr.name)[4]
            if number.isdigit() == False:
                number = os.path.basename(extr.name)[7]
            #if number.isdigit() == False:
                
            number = str(int(number)+1)
            images_folder=os.path.join(cam_folder,"cam"+number.zfill(2))
            #print("Loading images from:", images_folder)

            # Train: use every 3rd frame (1/3 of frames)
            # Test: use remaining 2/3 frames (frames not used in training)
            all_indices = list(range(image_length))
            train_indices = all_indices[::3]  # Every 3rd frame: 0, 3, 6, 9, ...
            test_indices = [i for i in all_indices if i not in train_indices]  # Remaining frames: 1, 2, 4, 5, 7, 8, ...

            if split == "train":
                image_range = train_indices
            elif split == "test":
                image_range = test_indices
            else:
                image_range = all_indices

            for i in image_range:
                #print(i)
                num=i+1
                image_path=os.path.join(images_folder,"cam_"+str(number).zfill(4)+'_'+str(num).zfill(4)+".jpg")
                #print(image_path)
                image_paths.append(image_path)
                image_poses.append((R,T))
                image_times.append(float(i/image_length))
        #print(image_paths)
        return image_paths, image_poses,image_times
    
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