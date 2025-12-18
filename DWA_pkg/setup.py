from setuptools import find_packages, setup

package_name = 'DWA_pkg'

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
            'dwa_node=DWA_pkg.dwa_2:main',
            'dwa_reduce_vel_node=DWA_pkg.dwa_reduce_vel:main',
        ],
    },
)
