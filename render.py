import bpy
import json
import os
import mathutils
import math
import time
import pickle
import pathlib
import glob
import subprocess
import shutil
import numpy as np
from time import sleep
from scipy.interpolate import interp1d
from mathutils import Euler, Vector, Quaternion, Matrix
from scipy.signal import savgol_filter

scene = bpy.data.scenes[0]
filepath = bpy.data.filepath
work_dir = os.path.dirname(filepath)

###################################################
# configurations
data_folder = os.path.join(work_dir, "src_data") # Set the name of data folder
render_dir = os.path.join(data_folder, "anim_out") # Set the name of render folder
target_file = '*'  # * for all files
data_selection = 'out'  # human or out
character_num = "123" 
cam_num = int(character_num)
resolution_percentage = 100
render_video = True  # True : render video
                     # False : render image frames only
upsample = True
out_fps = 30
verbose = True
test_run = False  # True : render first 10 frames for the test
###################################################

render = bpy.data.scenes[0].render
bone_info = {'Neck': 0,
             'Nose': 1, 
             'Head': 2, 
             'shoulder.R': 3, 
             'elbow.R': 4, 
             'wrist.R': 5, 
             'shoulder.L': 6, 
             'elbow.L': 7, 
             'wrist.L': 8}
             
def get_bone_vec(dir_vec, idx):
    return Vector((dir_vec[idx*3 + 0], -dir_vec[idx*3 + 2], -dir_vec[idx*3 + 1]))

def refine_elbow(shoulder_vec, elbow_vec):
    proj_mat = Matrix.OrthoProjection(shoulder_vec, 4).to_3x3()
    projected_vec = elbow_vec * proj_mat   
    mid_vector = (projected_vec + elbow_vec) / 2
    return mid_vector

def refine_nose(nose_vec):
    rest_vec = Vector((0, 0, 1))
    mid_vector = (rest_vec + nose_vec) / 2
    return mid_vector

def refine_spine(spine_vec):
    rest_vec = Vector((0, -0.1, 1))
    mid_vector = (rest_vec + spine_vec) / 2
    return mid_vector

def upsample(poses):
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses * 2, poses.shape[0]))
    for i in range(poses.shape[1]):
        f = interp1d(list(range(n_poses)), poses[:, i], kind='cubic')
        x_new = np.linspace(0, n_poses - 1, num=n_poses*2, endpoint=True)
        out_poses[:, i] = savgol_filter(f(x_new), 9, 3)
    return out_poses

def fetch_files(data_folder):
    pkl_paths = sorted(glob.glob((os.path.join(data_folder, "{}*.pkl".format(target_file)))))
    print(pkl_paths)
    return pkl_paths

def render(pkl_path):
    # load pkl
    pkl_data = pickle.load(open(pkl_path, "rb"))
    
    if data_selection == 'human':
        dir_vec = pkl_data['human_dir_vec']
    else:
        dir_vec = pkl_data['out_dir_vec']
    print(dir_vec[0].reshape(-1, 3))
    
    # upsample
    if upsample:
        dir_vec = upsample(dir_vec)
    
    # select charactor
    bpy.context.scene.objects.active = bpy.context.scene.objects.get("Armature."+str(character_num))
    bpy.ops.object.mode_set(mode='POSE')
    objName = "Armature."+str(character_num)
    bones = bpy.data.objects[objName].pose.bones
    
    # set frame Number
    n_frames = len(dir_vec)
    
    if test_run:
        n_frames = 10
    
    # make the rest pose
    bpy.context.active_object.animation_data_clear()
    rest_pose = {}
    
    for name in bones.keys():
        bones[name].rotation_quaternion = [1, 0, 0, 0]
    bpy.context.scene.update()
    
    # make keyframes
    for j in range(n_frames):
        shoulder_vec = Vector(bones['elbow.R'].head - bones['elbow.L'].head)
        for bone_name in bones.keys():
            if bone_name == 'Head':
                continue
            bone = get_bone_vec(dir_vec[j], bone_info[bone_name])
            if 'elbow' in bone_name:                
                bone = refine_elbow(shoulder_vec, bone)
            elif 'nose' in bone_name.lower():
                bone = refine_nose(bone)
            elif 'neck' in bone_name.lower():
                bone = refine_spine(bone)
            if verbose and j == 0:
                print(bone_name)
                print('\t', bone)
                print('\t', bone * bones[bone_name].matrix)
    
            bone = bone * bones[bone_name].matrix
            rot = Vector((0, 1, 0)).rotation_difference(bone)
    
            bones[bone_name].rotation_quaternion = bones[bone_name].rotation_quaternion * rot
            bones[bone_name].keyframe_insert(data_path='rotation_quaternion',frame=j)
            bpy.context.scene.update()
    
    ### Render Setting ###
    
    # resolution
    cam = "Camera." + str(cam_num)
    bpy.context.scene.camera = bpy.data.objects[cam]
    bpy.context.scene.render.resolution_x = 1280
    bpy.context.scene.render.resolution_y = 1280
    bpy.context.scene.render.resolution_percentage = resolution_percentage
    
    # set start/end frames
    scene.frame_start = 0
    scene.frame_end = n_frames
    scene.frame_step = 1
    
    # clear existing files
    img_path = os.path.join(render_dir, os.path.basename(pkl_path))
    shutil.rmtree(img_path, True)
    
    # render
    bpy.context.scene.render.filepath = os.path.join(img_path, "####.png")
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    
    start_time = time.time()
    scene.use_nodes = True
    
    bpy.context.scene.render.fps = out_fps
    bpy.ops.render.render(animation=True)
    elapsed_time = time.time() - start_time
    
    print('rendering time: {:.1f} s, {:.1f} FPS'.format(elapsed_time, n_frames / elapsed_time))
    
    if render_video:
        merge_video_audio(img_path, os.path.basename(pkl_path))
def merge_video_audio(img_path, name):
    
    if data_selection == 'human':
        merged_video_path = os.path.join(render_dir, name.replace('.pkl', '_human.mov'))
    else:
        merged_video_path = os.path.join(render_dir, name.replace('.pkl', '.mov'))
    
    img_seq = '{}/{}.png'.format(img_path, '%04d')
    files = glob.glob('{}/{}*.wav'.format(data_folder, os.path.splitext(name)[0]))
    assert len(files) > 0
    audio_path = files[0]
    
    print(audio_path,merged_video_path)
    cmd = ['ffmpeg', '-r', '30', '-y', '-i', img_seq, '-i', audio_path,
               '-vcodec', 'prores_ks', '-strict', '-2', merged_video_path]
    subprocess.call(cmd)

if __name__ == '__main__':
    pkl_list = fetch_files(data_folder)

    for pkl_filepath in pkl_list:
        print(pkl_filepath)
        render(pkl_filepath)
        # break