run:
	nvcc ./grad.cu -o grad.exe
	./grad.exe

build:
	nvcc ./grad.cu -o grad.exe