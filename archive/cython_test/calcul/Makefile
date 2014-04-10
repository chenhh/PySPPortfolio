CC=gcc 
CFLAGS=-Wall -O2 -fPIC
OBJECTS = calcul.o

lib: $(OBJECTS)
	$(CC) -fPIC -shared -o libcalcul.so  calcul.c
