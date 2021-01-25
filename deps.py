"""Routines to install torch geometric dependencies."""
import subprocess
import sys

if sys.version_info.minor < 8:
    PYTHON_VERSION = 'cp{major}{minor}-cp{major}{minor}m'.format(
        major=sys.version_info.major, minor=sys.version_info.minor)
else:
    PYTHON_VERSION = 'cp{major}{minor}-cp{major}{minor}'.format(
        major=sys.version_info.major, minor=sys.version_info.minor)

PLATFORM = "macosx_10_9_x86_64" if sys.platform == 'darwin' else "linux_x86_64"


WHEELS = [
    "https://pytorch-geometric.com/whl/torch-1.7.0/torch_cluster-latest+{cuda}-{python}-{platform}.whl",
    "https://pytorch-geometric.com/whl/torch-1.7.0/torch_scatter-latest+{cuda}-{python}-{platform}.whl",
    "https://pytorch-geometric.com/whl/torch-1.7.0/torch_sparse-latest+{cuda}-{python}-{platform}.whl",
    "https://pytorch-geometric.com/whl/torch-1.7.0/torch_spline_conv-latest+{cuda}-{python}-{platform}.whl"

]


def install_deps(cuda):
    pip_install_command = ['poetry', 'run', 'pip', 'install']
    for wheel in WHEELS:
        subprocess.call(pip_install_command + [wheel.format(
            cuda=cuda, python=PYTHON_VERSION, platform=PLATFORM
        )])


def install_deps_cpu():
    install_deps('cpu')


def install_deps_cu101():
    install_deps('cu101')


def install_deps_cu102():
    install_deps('cu102')


def install_deps_cu110():
    install_deps('cu110')
