from setuptools import setup
from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
import glob
import shutil


__library_file__ = './lib/*.so'
__version__ = '0.0.1'



class CopyLibFile(install):
    """"
    Directly copy library file to python's site-packages directory.
    """

    def run(self):
        install_dir = get_python_lib()
        lib_file = glob.glob(__library_file__)
        # assert len(lib_file) == 1     

        print('copying {} -> {}'.format(lib_file[0], install_dir))
        shutil.copy(lib_file[0], install_dir)

        print('copying {} -> {}'.format(lib_file[1], install_dir))
        shutil.copy(lib_file[1], install_dir)




setup(
    name='g2opy',
    version=__version__,
    description='g2o framework and pangolin',
    url='',
    license='BSD',
    cmdclass=dict(
        install=CopyLibFile
    ),
    keywords='g2o, pangolin',
    long_description="""
        g2o is an open-source C++ framework for optimizing graph-based nonlinear 
        error functions. g2o has been designed to be easily extensible to a wide 
        range of problems and a new problem typically can be specified in a few 
        lines of code. The current implementation provides solutions to several 
        variants of SLAM and BA.
        
        Pangolin is a lightweight portable rapid development library for managing 
        OpenGL display / interaction and abstracting video input. At its heart is 
        a simple OpenGl viewport manager which can help to modularise 3D visualisation 
        without adding to its complexity, and offers an advanced but intuitive 3D 
        navigation handler. Pangolin also provides a mechanism for manipulating 
        program variables through config files and ui integration, and has a 
        flexible real-time plotter for visualising graphical data."""
)