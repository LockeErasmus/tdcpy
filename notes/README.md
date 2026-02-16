# Notes

This is the place for putting internal stuff like

- propositions to improve some algorithm
- whitepapers not referenced in sphinx

## pyproject.toml

Consider adding to `[project.urls]`
```
Paper = "https://doi.org/xxxxx"
Preprint = "https://arxiv.org/abs/xxxx"
```

## Local Distribution build

Make sure `pip install build`, then

```
python -m build
```

builds, you should see `dist/*whl` and `dist/*tar.gz` files.
