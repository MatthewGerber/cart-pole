# cart-pole

# Installation
First SSH into the Raspberry Pi system.

Then clone the repository and configure a virtual environment. The following steps use 
[poetry](https://python-poetry.org/) for installation:

```shell
git clone --recurse-submodules git@github.com:MatthewGerber/cart-pole.git
cd cart-pole
poetry env use 3.11
poetry install
```

# To Do
* Check state-action tuple equality in reinforce first-visit check.
