# New features notes

We use Semantic versioning and we are still in initial development, The public
API can change. Breaking changes can appear, but we always bump minor version
`0.y.0` by 1.

## V0.1.0

 - get rid of all `assert` statements where it should be `raise`

### High Level API

 - consider `if isinstance(NDDE)` --> change to is not instance DDAE 

#### sa, cd, strong_sa

 - `return_info` kwarg (think of strong_sa, return info of rmr sa)

#### controller design

 ???

### Metadata structures

 - move all from `namedTuple` to `dataclass`

 

