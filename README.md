# cart-pole

# Installation
First SSH into the Raspberry Pi system.

Then clone the repository and configure a virtual environment. The following steps use 
[poetry](https://python-poetry.org/) for installation. You might need the following prior to installing:

```shell
export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
```

To use rotary encoders via the Arduino interface:
1. Disable the serial command line by editing `/boot/firmware/cmdline.txt` to remove `console=serial0,115200`.
2. Add `enable_uart=1` to `/boot/firmware/config.txt`.
3. Use `sudo raspi-config` to disable the serial shell and enable UART.
4. Reboot.

```shell
git clone --recurse-submodules git@github.com:MatthewGerber/cart-pole.git
cd cart-pole
poetry env use 3.11
poetry install
```

# To Do/Try
* Rotary encoder losses.
* Add time series database/dashboard.
* Move motor speed control to Arduino, including next-set promise.
* Add serial read/write throughput.
* Add I2C and serial devices to CLI args.
* Pass phase-change mode and phase-changes per rotation to Arduino rotary encoder.
* Resume latest checkpoint if file not specified.
* Shouldn't need "--save-agent-path" in resume.
* Check state-action tuple equality in reinforce first-visit check.
