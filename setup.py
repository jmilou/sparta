from setuptools import setup, find_packages

setup(
    name='sparta',
    version='0.1',

    description='Miscellaneous Python utilities to analyze SPHERE sparta images and retrieve ambient conditions informations',
    url='https://github.com/jmilou/sparta',
    author='Julien Milli',
    author_email='jmilli@eso.org',
    license='MIT',
    keywords='image processing data analysis',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'astropy', 'pandas', 'matplotlib','pandas','datetime'
    ],
    zip_safe=False
)
