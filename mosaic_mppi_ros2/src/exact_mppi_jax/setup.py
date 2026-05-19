from setuptools import setup
import glob
import sys

# Colcon may invoke setup.py with flags setuptools doesn't recognize.
# Strip them so builds don't fail (e.g. with --symlink-install).
UNSUPPORTED_VALUE_FLAGS = {"--build-directory"}
UNSUPPORTED_SIMPLE_FLAGS = {"--editable", "-e"}

cleaned_argv = []
skip_next = False
for i, arg in enumerate(sys.argv):
    if skip_next:
        skip_next = False
        continue
    if arg in UNSUPPORTED_SIMPLE_FLAGS:
        continue
    if arg in UNSUPPORTED_VALUE_FLAGS:
        if i + 1 < len(sys.argv):
            skip_next = True
        continue
    cleaned_argv.append(arg)

sys.argv = cleaned_argv

package_name = 'exact_mppi_jax'

launch_files = glob.glob('launch/*.py')
config_files = glob.glob('config/*.yaml')
planner_config_files = glob.glob('config/mppi_config/*')
real_map_files = glob.glob('config/real_map/*')
rviz_files = glob.glob('rviz/*.rviz')
world_files = glob.glob('worlds/*')
model_files = [p for p in glob.glob('models/pedestrian_obstacle/*')
               if not (p.endswith('/meshes') or p.endswith('/materials'))]
mesh_files = glob.glob('models/pedestrian_obstacle/meshes/*')
texture_files = glob.glob('models/pedestrian_obstacle/materials/textures/*')
script_files = glob.glob('models/pedestrian_obstacle/materials/scripts/*')

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', launch_files),
        ('share/' + package_name + '/config', config_files),
        ('share/' + package_name + '/config/mppi_config', planner_config_files),
        ('share/' + package_name + '/config/real_map', real_map_files),
        ('share/' + package_name + '/rviz', rviz_files),
        ('share/' + package_name + '/worlds', world_files),
        ('share/' + package_name + '/models/pedestrian_obstacle', model_files),
        ('share/' + package_name + '/models/pedestrian_obstacle/meshes', mesh_files),
        ('share/' + package_name + '/models/pedestrian_obstacle/materials/textures', texture_files),
        ('share/' + package_name + '/models/pedestrian_obstacle/materials/scripts', script_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kevinlad',
    maintainer_email='kevinladlee@gmail.com',
    description='ROS2 bridge package for the installed exact_mppi JAX controller',
    license='GPL-3.0',
    keywords=['ROS'],
    entry_points={
        'console_scripts': [
            'mppi_local = exact_mppi_jax.mppi_local:main',
            'global_ref_path = exact_mppi_jax.global_ref_path:main',
            'global_ref_path_node = exact_mppi_jax.global_ref_path_node:main',
            'cost_breakdown_viz = exact_mppi_jax.cost_breakdown_viz:main',
            'cmd_vel_watchdog = exact_mppi_jax.cmd_vel_watchdog:main',
            'gazebo_dynamic_obstacles = exact_mppi_jax.gazebo_dynamic_obstacles:main',
            'corridor_goal_trigger = exact_mppi_jax.corridor_goal_trigger:main',
            'start_obstacles_on_goal = exact_mppi_jax.start_obstacles_on_goal:main',
        ],
    },
)
