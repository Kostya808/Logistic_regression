All: clean main

main:
	nvcc -std=c++11 -lcurand -lm main.cu -o main 

clean:
	rm -f main

