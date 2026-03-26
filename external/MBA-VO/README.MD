# MBA-VO
* Working in progress...

## Run C++

```bash
cmake -B build
cmake --build build
./build/bin/test_blur_aware_tracker_sequential
```

## Run python

```bash
cmake -B build
cmake --build build -t install
cd python
python setup.py install
python test/tracker.py
```
