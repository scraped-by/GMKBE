#!/usr/bin/env bash
if [ -e CMakeLists.txt ]; then
	cmake . && make
fi

if [ -e compile_commands.json ]; then
	sed -i 's/\/mnt\/c/C:/g' compile_commands.json
fi

