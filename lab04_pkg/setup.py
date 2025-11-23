from setuptools import find_packages, setup
from glob import glob

package_name = 'lab04_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aumoza',
    maintainer_email='aumoza2000@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ekf_node = lab04_pkg.ekf_node:main',
            'ekf_node_1 = lab04_pkg.ekf_node_1:main',
            'controller_burger_2 = lab04_pkg.controller_burger_2:main',
            'ekf_plot = lab04_pkg.ekf_plot:main',
        ],
    },
)
