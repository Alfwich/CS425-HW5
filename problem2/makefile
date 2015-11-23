CC = nvcc
OFLAGS = -O3

a.x: main.cu support_code.cpp
	$(CC) $(OFLAGS) main.cu support_code.cpp -o a.x

clean:
	rm -f a.x a.exp a.lib
