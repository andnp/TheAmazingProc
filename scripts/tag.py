import subprocess as sp
from typing import NamedTuple
import re

Version = NamedTuple('Version', [
    ('major', int),
    ('minor', int),
    ('patch', int),
])

# take a string like: '1.2.3'
# and return a Version(major=1, minor=2, patch=3)
def toVersion(s: str) -> Version:
    major, minor, patch = s.split('.')
    return Version(int(major), int(minor), int(patch))

def versionGreater(v1: Version, v2: Version) -> bool:
    return v1.major > v2.major or v1.minor > v2.minor or v1.patch > v2.patch

def versionEqual(v1: Version, v2: Version) -> bool:
    return v1.major == v2.major and v1.minor == v2.minor and v1.patch == v2.patch

# take a string specifying a command-line command + arguments
# returns the stdout output assuming returncode==0
def run(cmd: str):
    out = sp.run(cmd.split(' '), stdout=sp.PIPE, stderr=sp.PIPE)

    # assert process ended successfully
    out.check_returncode()

    # get stdout if so
    stdout = out.stdout.decode('utf8')

    # remove trailing new line that's usually used for the cli
    if stdout.endswith('\n'):
        return stdout[:-1]

    return stdout

# get latest git tag number
version_str = run('git describe --abbrev=0 --tags')
git = toVersion(version_str)

# get latest setup.py version
with open('setup.py', 'r') as f:
    setup_file = f.read()

# check if setup.py specifies a version number
match = re.search(r"\W*version='(.+?)',.*", setup_file, re.MULTILINE)
if match is not None:
    version_str = match.group(1)
    setup = toVersion(version_str)

# if not, default to 0
else:
    setup = Version(0, 0, 0)

# -----------------------------------
# -- Check which version is bigger --
# -----------------------------------
next_version = Version(0, 0, 1)
if versionEqual(git, setup):
    next_version = Version(git.major, git.minor, git.patch + 1)

elif versionGreater(setup, git):
    next_version = setup

else:
    # note this condition is pretty weird, the git version _generally_ shouldn't be bigger...
    next_version = git
