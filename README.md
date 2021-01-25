# Topological Graph Neural Networks

## Installation

Install using `poetry`:

```bash
$ poetry install
```

Afterwards install the dependencies of `torch_geometric` dependent on you cuda
version via

```bash
$ poetry run install_deps_{cpu, cu101, cu102, cu110}
```

where `{cpu, cu101, cu102, cu110}` should be replaced with either `cpu` or the
string matching the installed cuda toolkit version.
