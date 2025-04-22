# Shaft

This is a rough start for working with SHApe Fourier Transforms.

## What we got
Only a **playground** for now. Inside it are two folders:
- **curves** has:
  - *fourier.py*: A fourier implementation (it's just one function) applied to 2 triangulated curves. The triangulations are also done in there (it's also just one function).
  - *preprocessing.py*: This is just to extract the 2d curves from a csv into a numpy format. This has been runned already so no need to re-run.
- **surfaces** has:
  - Nothing related to fourier yet. Just the function to tetrahedralize a surface.

The **data** folder is just where the data used in the script are used (of course).

## Installation
There are no modules yet just random scripts, all you need is to clone the repo and install the requirements in pyproject.toml.

```
git clone https://github.com/djoharr/shaft
cd shaft
pip install -e .
```
