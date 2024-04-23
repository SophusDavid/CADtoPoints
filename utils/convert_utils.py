import numpy as np

def normalize(v):
    """标准化向量"""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def look_at(eye, center, up=[0, 1, 0]):
    """构造观察矩阵"""
    F = np.array(center) - np.array(eye)
    f = normalize(F)
    u = normalize(np.array(up))
    s = normalize(np.cross(f, u))
    u = np.cross(s, f)
    
    # 构建观察矩阵
    M = np.identity(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -np.array(eye)
    M[3, :3] = [0.0, 0.0, 0.0]
    M[3, 3] = 1.0
    
    return M

def convert_to_nerf_format(eye, center, up=[0, 1, 0]):
    """将相机参数转换为NeRF格式"""
    transform_matrix = look_at(eye, center, up)
    
    # NeRF格式需要的是世界到相机的转换矩阵，而look_at提供的是相机到世界的转换，所以取逆
    transform_matrix = np.linalg.inv(transform_matrix)
    
    return transform_matrix
def calculate_fov_x(fov_y_degrees, aspect_ratio):
    # 将FOV Y从度转换为弧度
    fov_y_radians = np.deg2rad(fov_y_degrees)
    
    # 计算FOV X
    fov_x_radians = 2 * np.arctan(np.tan(fov_y_radians / 2) * aspect_ratio)
    
    # 将FOV X从弧度转换回度
    # fov_x_degrees = np.rad2deg(fov_x_radians)
    
    return fov_x_radians
def calculate_fov_y(fov_x_degrees, aspect_ratio):
    # 将FOV X从度转换为弧度
    fov_x_radians = np.deg2rad(fov_x_degrees)
    # 计算FOV Y
    fov_y_radians = 2 * np.arctan(np.tan(fov_x_radians / 2) / aspect_ratio)
    # 将FOV Y从弧度转换回度
    # fov_y_degrees = np.rad2deg(fov_y_radians)
    return fov_y_radians

    
    
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),
    #     image_width=int(viewpoint_camera.image_width),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform,
    #     projmatrix=viewpoint_camera.full_proj_transform,
    #     sh_degree=pc.active_sh_degree,
    #     campos=viewpoint_camera.camera_center,
    #     prefiltered=False,
    #     debug=pipe.debug
    # )
def slerp(p0, p1, t):
    """球面线性插值"""
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

# 根据训练集的位姿生成一个平滑的位姿序列用来渲染视频
def generate_smooth_poses(poses, num_frames, smoothness, num_keyframes):
    """
    :param poses: 位姿列表
    :param num_frames: 生成的位姿数量
    :param smoothness: 平滑度
    :param num_keyframes: 关键帧数量
    :return: 平滑的位姿列表
    """
    # 计算关键帧之间的间隔
    interval = len(poses) // num_keyframes
    # 选取关键帧
    keyframe_poses = poses[::interval]
    # 生成平滑的位姿序列
    smooth_poses = []
    for i in range(len(keyframe_poses) - 1):
        # 计算两个关键帧之间的位姿数量
        num_sub_frames = num_frames // num_keyframes
        # 生成两个关键帧之间的平滑位姿序列
        for j in range(num_sub_frames):
            # 计算权重
            w = smoothness(j / num_sub_frames)
            # 线性插值
            smooth_poses.append(slerp(keyframe_poses[i], keyframe_poses[i + 1], w))
    # 添加最后一个关键帧
    smooth_poses.append(keyframe_poses[-1])
    return smooth_poses
