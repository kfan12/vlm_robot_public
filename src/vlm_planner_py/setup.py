from setuptools import setup

package_name = 'vlm_planner_py'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/vlm_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/vlm_planner.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='VLM planner for simulated robotcar',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vlm_node = vlm_planner_py.vlm_node:main',
            # NOTE: vlm_sign_node needs the torch venv. `ros2 run` uses the
            # console-script's system-python shebang (no torch), so launch it with
            # the venv python instead:  python3 -m vlm_planner_py.vlm_sign_node
            'vlm_sign_node = vlm_planner_py.vlm_sign_node:main',
        ],
    },
)
