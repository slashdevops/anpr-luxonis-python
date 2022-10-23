from subprocess import call

import pkg_resources

for dist in pkg_resources.working_set.require():
    print(f"updating -> {dist.project}")
    call("python -m pip install --upgrade " + dist.project, shell=True)
