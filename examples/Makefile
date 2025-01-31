CC = gcc
CFLAGS = -shared -fPIC
LIBS = -lm -lgsl -lgslcblas
TARGET = monte_carlo_integral.so
SRC = monte_carlo_integral.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)