# tdspy


## Installation

### Installing with `pip`

From github
```bash
pip install tdspy@git+https://github.com/LockeErasmus/tdspy
```

Local install
```bash
pip install <path-to-tdspy>
```

### Installing from source

Clone repository
```bash
git clone https://github.com/LockeErasmus/tdspy.git
```

install from source with `-e` if you are in repository
```bash
pip install -e .
```
or
```bash
pip install -e <path-to-tdspy>
```
otherwise.

## Testing

As of now it is usefull to run tests with visible outputs:
```
pytest -rP
```
shows output of passed tests,
```
pytest -rx
```
shows output of failed tests (default behaviour of pytest)




## Usage notes

### Coming from matlab

Original `tds-control` MATLAB package constructed matrices as structure `A={A0, A1, ...}`, it is more computationally efficent to leverage C code (use as much numpy and as little pure python as possible), i.e.

```python
import numpy as np

# matlab-like code
A0 = np.array([[-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -10, -4],
                [0, 0, 4, -10]])
A1 = np.array([[3, 3, 3, 3],
            [0, -1.5, 0, 0],
            [0, 0, 3, -5],
            [0, 5, 5, 5]])
A = [A0, A1]
hA = [0, 1]

# better in python
A = np.stack([A0, A1], axis=2) # results in array (4,4,2)
hA = np.array([0., 1])
```