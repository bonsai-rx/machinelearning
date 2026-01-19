import sys
import os
import subprocess
import argparse

def get_venv_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '.venv')

def get_pip_path(venv_path: str = None):
    if venv_path is None:
        venv_path = get_venv_path()
    if sys.platform.startswith('linux'):
        return os.path.join(venv_path, 'bin', 'pip')
    else:
        return os.path.join(venv_path, 'Scripts', 'pip.exe')
    
def get_python_path(venv_path: str = None):
    if venv_path is None:
        venv_path = get_venv_path()
    if sys.platform.startswith('linux'):
        return os.path.join(venv_path, 'bin', 'python')
    else:
        return os.path.join(venv_path, 'Scripts', 'python.exe')

def get_base_dir(base_dir: str = None):
    # function to get the base directory
    if base_dir is not None:
        return base_dir
    try:
        return os.path.dirname(os.path.realpath(__file__))
    except:
        return os.getcwd()

def create_venv(parent_dir: str = None):
    # function to create a virtual environment
    if parent_dir is None:
        parent_dir = os.path.dirname(os.path.realpath(__file__))
    venv_path = os.path.join(parent_dir, ".venv")
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])
    return venv_path

def activate_venv(venv_path: str = None):
    # function to activate the virtual environment
    if venv_path is None:
        venv_path = get_venv_path()
    if sys.platform.startswith('linux'):
        bin_path = os.path.join(venv_path, 'bin')
        os.environ["PATH"] = os.pathsep.join([bin_path, *os.environ.get("PATH", "").split(os.pathsep)])
        sys.path.insert(0, os.path.join(venv_path, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'))
    else:
        bin_path = os.path.join(venv_path, 'Scripts')
        os.environ["PATH"] = os.pathsep.join([bin_path, *os.environ.get("PATH", "").split(os.pathsep)])
        sys.path.insert(0, os.path.join(venv_path, 'Lib', 'site-packages'))

def install(venv_path: str = None, pip_args: list[str] = None):
    # function to install pip packages into a virtual environment
    if venv_path is None:
        venv_path = get_venv_path()
    pip_path = get_pip_path(venv_path)
    if pip_args is None:
        raise ValueError("pip_args must be provided")
    subprocess.check_call([pip_path, "install", *pip_args])

def install_requirements(requirements_file: str, venv_path: str = None):
    # function to install pip packages from a requirements file into a virtual environment
    if venv_path is None:
        venv_path = get_venv_path()
    pip_path = get_pip_path(venv_path)
    subprocess.check_call([pip_path, "install", "-r", requirements_file])

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str, default=None)
args = parser.parse_args()

base_dir = get_base_dir(args.base_dir)
venv_path = create_venv(base_dir)
activate_venv(venv_path)

install(venv_path, ["--no-cache-dir", "torch"])
install(venv_path, ["--no-cache-dir", "plotly"])
install(venv_path, ["--no-cache-dir", "remfile"])
install(venv_path, ["--no-cache-dir", "dandi"])
install(venv_path, ["--no-cache-dir", "ssm@git+https://github.com/ncguilbeault/lds_python@75e3e5e92ce6344009b62a5034db49b238db63ef"])

python_path = get_python_path(venv_path)

script_path = os.path.join(base_dir, "estimate_neural_latents.py")
process = subprocess.Popen([python_path, script_path, base_dir])
return_code = process.wait()

if return_code == 0:
    print("Script completed successfully.")
else:
    print(f"Script exited with errors. Return code: {return_code}")