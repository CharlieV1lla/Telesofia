### Task parameters

DATA_DIR = '/home/carlos/recorded_data'
TASK_CONFIGS = {
    'touch_side_table':{
        'dataset_dir': DATA_DIR + '/touch_side_table',
        'num_episodes': 20,
        'episode_len': 1000,
        'camera_names': None
    },
        'touch_side_table_img':{
        'dataset_dir': DATA_DIR + '/touch_side_table_img',
        'num_episodes': 5,
        'episode_len': 1000,
        'camera_names': None
    },
    'say_yes':{
        'dataset_dir': DATA_DIR + '/say_yes',
        'num_episodes': 5,
        'episode_len': 1000,
        'camera_names': None
    },
        'look_at_lid':{
        'dataset_dir': DATA_DIR + '/look_at_lid',
        'num_episodes': 20,
        'episode_len': 1000,
        'camera_names': ['cam_table']
    },
        'look_at_cube':{
        'dataset_dir': DATA_DIR + '/look_at_cube',
        'num_episodes': 20,
        'episode_len': 1000,
        'camera_names': ['cam_table']
    },
        'look_at_cube_small':{
        'dataset_dir': DATA_DIR + '/look_at_cube_small',
        'num_episodes': 20,
        'episode_len': 1000,
        'camera_names': ['cam_table']
    },
        'look_at_cube_small_2_demos':{
        'dataset_dir': DATA_DIR + '/look_at_cube_small',
        'num_episodes': 2,
        'episode_len': 1000,
        'camera_names': ['cam_table']
    },
        'look_at_cube_small_90':{
        'dataset_dir': DATA_DIR + '/look_at_cube_small_90',
        'num_episodes': 20,
        'episode_len': 1000,
        'camera_names': ['cam_table']
    },
        'look_at_cube_small_RGBD':{
        'dataset_dir': DATA_DIR + '/look_at_cube_small_RGBD',
        'num_episodes': 20,
        'episode_len': 500,
        'camera_names': ['color'],
        'depth': True
    },
        'push_cube':{
        'dataset_dir': DATA_DIR + '/push_cube',
        'num_episodes': 54,
        'episode_len': 1000,
        'camera_names': ['color'],
        'depth': True
    },
        'push_cube_2_init':{
        'dataset_dir': DATA_DIR + '/push_cube_2_init',
        'num_episodes': 20,
        'episode_len': 1000,
        'camera_names': ['color'],
        'depth': True
    },
        'touch_center_table':{
        'dataset_dir': DATA_DIR + '/touch_center_table',
        'num_episodes': 10,
        'episode_len': 1000,
        'camera_names': ['color'],
        'depth': True
    },
    'push_cube_1p':{
        'dataset_dir': DATA_DIR + '/push_cube_1p',
        'num_episodes': 12,
        'episode_len': 1000,
        'camera_names': ['color'],
        'depth': True
    },
        'push_cube_2p':{
        'dataset_dir': DATA_DIR + '/push_cube_2p',
        'num_episodes': 8,
        'episode_len': 1000,
        'camera_names': ['color'],
        'depth': True
    },

        'push_cube_left_right':{
        'dataset_dir': DATA_DIR + '/push_cube_left_right',
        'num_episodes': 20,
        'episode_len': 750,
        'camera_names': ['color'],
        'depth': True
    },
}

### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
