# Cookbook for me to create documentation

Assuming we are at root, make sure we installed all `sphinx` dependencies

```
python -m pip install -r ./docs/requirements.txt
```

```
cd docs
```

```
sphinx-build -b html source build/html
```


## Starting from scratch (DO NOT USE)

```
sphinx-quickstart
```

Genereate `.rst` files for modules

```
sphinx-apidoc -o ./source/ ../src/tdspy/
```

Option if it won't create `.rst` files add `... --force --separate`

