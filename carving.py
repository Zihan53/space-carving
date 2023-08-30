import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

# def load_cameras(num, cam_prefix):
#     cams = []

#     for i in range(1, num):
#         cam = []
#         with open(f"{cam_prefix}{i:04}.txt" , 'r') as f:
#             lines = f.readlines()
#             for line in lines[1:4]:
#                 cam.append([float(x) for x in line.split()])
#         cams.append(np.array(cam))

#     return cams

def load_cameras(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_images = int(lines[0])
    cams = []
    for line in lines[1:num_images+1]:
        par = np.array([float(x) for x in line.split()[1:]])
        K = par[0:9].reshape(3, 3)
        R = par[9:18].reshape(3, 3)
        T = par[18:].reshape(3, 1)
        P = K @ np.hstack([R, T])
        #cams.append({"K":K, "R":R, "T":T})
        cams.append(P)

    return num_images, cams

def load_images(num, image_prefix):
    images = []

    for i in range(1, num):
        image_file = f"{image_prefix}{i:04}.png" 
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        images.append(image)

        # plt.imshow(image, cmap='gray')
        # plt.show()
        # break

    return images

def create_voxel_grids(size):
    x, y, z = np.mgrid[size[0]:size[1]+1, size[2]:size[3]+1, size[4]:size[5]+1]
    voxel_grids = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    voxel_grids = voxel_grids / 2000
    return voxel_grids


def space_carving(images, cameras, size):
    voxel_grids = create_voxel_grids(size)
    counts = []
    width = images[0].shape[1]
    height = images[0].shape[0]
    i = 0
    for image, camera in zip(images, cameras):
        print(i)
        i += 1
        # Project to 2D
        uvs = np.dot(np.vstack((voxel_grids, np.ones(voxel_grids.shape[1]))).T, camera.T)
        uvs = uvs[:, :2] / uvs[:, 2].reshape(-1, 1)
        # Find indice that are inside the imagex
        inside_image = np.where((0 <= uvs[:, 0]) & (uvs[:, 0] < width) & (0 <= uvs[:, 1]) & (uvs[:, 1] < height))[0]
        print(len(inside_image))
        # Find indice that are in silhouette
        silhouette = np.where(image[uvs[inside_image, 1].astype(int), uvs[inside_image, 0].astype(int)] > 48)[0]
        print(len(silhouette))
        # Record
        count = np.zeros((uvs.shape[0]))
        count[inside_image[silhouette]] = 1
        counts.append(count)

    count_sum = np.sum(counts, axis=0)
    # print(count_sum.shape)
    model = voxel_grids[:, count_sum > 277]
    return model.T

def visualize(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector([[0, 0, 139] for _ in range(len(points))])
    o3d.visualization.draw_geometries([pcd])

# num = 21
num, cams = load_cameras("./dino/dino_par.txt")
images = load_images(num, "./dino/dino")
model = space_carving(images, cams, [-84, 62, 4, 178, -76, 72])
visualize(model)