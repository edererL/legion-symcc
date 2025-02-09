CLANG = clang
COMPILER_H   = $(wildcard compiler/*.h)
COMPILER_CPP = $(wildcard compiler/*.cpp)

RUNTIME_H    = $(wildcard runtime/*.h)
RUNTIME_CPP  = $(wildcard runtime/*.cpp)

.PHONY: all docker

all: lib/libSymbolize.so lib/libSymRuntime.so lib32/libSymRuntime.so

fp: fp.c
	gcc fp.c -o fp -lz3 -Wall

lib/libSymRuntime.so: lib $(RUNTIME_CPP) $(RUNTIME_H)
	$(CLANG) -std=c++17 -Wall $(RUNTIME_CPP) -Iruntime -fPIC -shared -o $@

lib32/libSymRuntime.so: lib32 $(RUNTIME_CPP) $(RUNTIME_H)
	$(CLANG) -std=c++17 -Wall $(RUNTIME_CPP) -Iruntime -fPIC -shared -o $@ -m32

lib/libSymbolize.so: lib $(COMPILER_CPP) $(COMPILER_H)
	$(CLANG) -std=c++17 -Wall $(COMPILER_CPP) -Wall -fPIC -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -Wl,-z,nodelete -shared -o $@

lib:
	mkdir -p lib

lib32:
	mkdir -p lib32

README.html: README.md
	pandoc $< -s -o $@

docker: lib/libSymbolize.so lib/libSymRuntime.so lib32/libSymRuntime.so
	docker build . -t gidonernst/legion-symcc
	./docker-cp.sh

archive:
	rm legion/__pycache__ -rf
	mkdir -p ../testcomp-archives-2023/2023/legion-symcc
	cp -r legion.sh Legion.py Verifier.cpp legion lib lib32 dist LICENSE LICENSE.symcc LICENSE.z3 ../testcomp-archives-2023/2023/legion-symcc
	(cd ../testcomp-archives-2023/2023/; zip legion-symcc.zip legion-symcc/ -r)