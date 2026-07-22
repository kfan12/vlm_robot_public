from setuptools import find_packages, setup

package_name = 'robotcar_utils_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kefan',
    maintainer_email='tju.fanke@gmail.com',
    description='Python utility nodes: odom_path_rezero and path_drawer.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'path_drawer = robotcar_utils_py.path_drawer_node:main',
            'odom_path_rezero = robotcar_utils_py.odom_path_rezero:main',
            'teleop_keyboard = robotcar_utils_py.teleop_keyboard:main',
            'cone_markers = robotcar_utils_py.cone_marker_node:main',
            'fpv_truth_overlay = robotcar_utils_py.fpv_truth_overlay:main',
        ],
    },
)
