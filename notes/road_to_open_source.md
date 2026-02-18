# Road to opensource

## Important to ask Wim

- [x] Ask Wim if he is even ok with it - **Wim is ok**

    this project started as my private personal work to better understand
    algorithms I am using and also because I do not like paywalls (Hello MATLAB). I originaly had no intensions to make it public, but then some other guys started using it and apparently liked it ...

- [x] How to reference Wim's original work?

    - README.md - reference gitlab, based on tds-control
    - in documentation, reference selectively per example
    - Diego proposed to look at how PyGRANSO is done: https://ncvx.org/index.html

- [ ] How to disclaim, this is opensource project not connected to Wim (in the sense that Wim would be responsible for bugs, etc...)
- [x] Decide on lincense - Just go with standard copyleft license GNU GPLv3 license


## Documentation

- [ ] `README.md`
- [ ] docstrings with emphasis on Sphinx
    - [] **ALL docstrings into NumpyDoc format**
- [ ] documentation hosting
    - candidates: https://readthedocs.org/ or https://pages.github.com/
- [ ] automatic build
- [ ] `.md` document for coming from matlab
- [ ] some kind of reference to this work (tds-control MANUAL could be inspiration)

- [] Examples gallery (see https://sphinx-gallery.github.io/stable/auto_examples/index.html)
    - Animation example https://sphinx-gallery.github.io/stable/auto_examples/plot_8_animations.html for discretization

## PyPI

- [ ] Decide on PyPI name (`tdcpy` is taken sadly)

1. tddpy - time delay dynamics
1. tdcpy - time delay control
1. tdapy - time delay analysis
1. tdscpy - time delay systems control

## Examples

## Testing

- [ ] unit tests for API
- [ ] Ensure tests pass on all supported Python versions

## Contributing
- [ ] Add **CONTRIBUTING.md** with:
    - How to report issues
    - How to submit pull requests
    - Code style guidelines


