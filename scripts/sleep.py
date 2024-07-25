from interbotix_xs_modules.arm import InterbotixManipulatorXS
from robot_utils import move_arms, torque_on

def main():
    puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet', init_node=True)
    master_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'master', init_node=False)

    all_bots = [puppet_bot, master_bot]
    for bot in all_bots:
        torque_on(bot)

    sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    move_arms(all_bots, [sleep_position] * 2, move_time=2)

if __name__ == '__main__':
    main()
