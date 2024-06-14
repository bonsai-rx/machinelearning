import subprocess
import sys
import os
import argparse

def get_base_dir(base_dir = None):
    # function to get the base directory
    if base_dir is not None:
        return base_dir
    try:
        return os.path.dirname(os.path.realpath(__file__))
    except:
        return os.getcwd()

def create_venv(parent_dir = None):
    # function to create a virtual environment
    if parent_dir is None:
        parent_dir = os.path.dirname(os.path.realpath(__file__))
    venv_path = os.path.join(parent_dir, ".venv")
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])
    return venv_path

def activate_venv(venv_path = None):
    # function to activate the virtual environment
    if venv_path is None:
        venv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.venv')
    if sys.platform.startswith('linux'):
        bin_path = os.path.join(venv_path, 'bin')
        os.environ["PATH"] = os.pathsep.join([bin_path, *os.environ.get("PATH", "").split(os.pathsep)])
        sys.path.insert(0, os.path.join(venv_path, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'))
    else:
        bin_path = os.path.join(venv_path, 'Scripts')
        os.environ["PATH"] = os.pathsep.join([bin_path, *os.environ.get("PATH", "").split(os.pathsep)])
        sys.path.insert(0, os.path.join(venv_path, 'Lib', 'site-packages'))

def install(package, venv_path = None):
    # function to install pip packages into a virtual environment
    if venv_path is None:
        venv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.venv')
    if sys.platform.startswith('linux'):
        pip_path = os.path.join(venv_path, 'bin', 'pip')
    else:
        pip_path = os.path.join(venv_path, 'Scripts', 'pip.exe')
    subprocess.check_call([pip_path, "install", package])

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=str, default=None)
parser.add_argument("n_samples", type=int, default=1)
args = parser.parse_args()

base_dir = get_base_dir(args.base_dir)
venv_path = create_venv(base_dir)
activate_venv(venv_path)
install("pandas", venv_path)
install("lds_python@git+https://github.com/ncguilbeault/lds_python@dc7a2e02892033734746bc0615dc294f5b43f672", venv_path)

if sys.platform.startswith('linux'):
    python_path = os.path.join(venv_path, "bin", "python")
else:
    python_path = os.path.join(venv_path, "Scripts", "python.exe")

script_path = os.path.join(base_dir, "receptive_field.py")
process = subprocess.Popen([python_path, script_path, base_dir, str(args.n_samples)])
return_code = process.wait()

if return_code == 0:
    print("Script completed successfully.")
else:
    print(f"Script exited with errors. Return code: {return_code}")