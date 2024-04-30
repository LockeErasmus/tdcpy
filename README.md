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

install from source with `-e`
```bash
pip install -e tdspy
```


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