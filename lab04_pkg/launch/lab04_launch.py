from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    ekf_node = Node(
        package= 'lab04_pkg',
        executable = 'ekf_node',
        name = 'ekf_node',
        output = 'screen',
    )
    #  controllers_node = Node(
    #     package= 'lab04_pkg',
    #     executable = 'controllers_node',
    #     name = 'controllers_node',
    #     output = 'screen',
    # )
    
    ekf_plot_node = Node(
        package= 'lab04_pkg',
        executable = 'ekf_plot',
        name = 'ekf_plot_node',
        output = 'screen',
    )

    return LaunchDescription([
        ekf_node,
        ekf_plot_node
    ])
