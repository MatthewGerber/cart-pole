# cart-pole

# Installation
First SSH into the Raspberry Pi system.

Then clone the repository and configure a virtual environment. The following steps use 
[poetry](https://python-poetry.org/) for installation. You might need the following prior to installing:

```shell
export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
```

```shell
git clone --recurse-submodules git@github.com:MatthewGerber/cart-pole.git
cd cart-pole
poetry env use 3.11
poetry install
```

# To Do/Try
* Slack in belt causes cart state to drift such that the cart is actually farther left than the state indicates, which
*   makes sense since there is more slack when pulling the cart right, and our clockwise indicator is more often incorrect 
*   in that scenario being driven by the motor speed rather than direct observation.
* Need to lock shared-memory Value objects.
* Hide dual rotary encoders underlying the DMP class.
* Monitor multiprocessing for killed processes.
* Resume latest checkpoint if file not specified.
* Shouldn't need "--save-agent-path" in resume.
* Check state-action tuple equality in reinforce first-visit check.
