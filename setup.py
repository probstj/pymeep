from setuptools import setup

setup(name='pymeep',
      version='0.1',
      description='Minimal python package to run meep simulations',
      url='http://github.com/probstj/pymeep',
      author='Jürgen Probst',
      author_email='juergen.probst@gmail.com',
      license='GPLv3',
      packages=['pymeep'],
      install_requires=[
          'numpy', 
          'matplotlib',
      ],
      extras_require={
          'gds':  ["gdspy>=1.1"],
      },
      zip_safe=False)
