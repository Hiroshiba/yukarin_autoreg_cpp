# yukarin_autoreg_cpp

## Prepare
```bash
git clone https://github.com/NVIDIA/cub
```

## Build
```bash
# make clean
make EXTRA_NVCCFLAGS="-I./cub/cub" EXTRA_CCFLAGS="-fPIC"
```
