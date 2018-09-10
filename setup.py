from setuptools import setup, find_packages

setup(name='pydygp',
      version='0.2.0.dev1',
      author='Daniel Tait',
      author_email='tait.djk@gmail.com',
      url='http://github.com/danieljtait/pydygp',
      license='MIT',
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib>=2.2.0',
      ],
      zip_safe=False)      
