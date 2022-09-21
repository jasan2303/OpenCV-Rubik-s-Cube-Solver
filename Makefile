INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
CFLAGSNO= -O0 -g $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt
LIBSNO= -L/usr/lib -lrt
HFILES= rubiks.h
CFILES= main.cpp rubiks.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all: main 

clean:
	-rm -f *.o *.d *.png
	-rm -f main

#rubiks: rubiks.cpp
#	g++ -o rubiks rubiks.cpp

#main: main.cpp
#	$(CC) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)
#rubiks: $(OBJS)
#	$(CC) $(LDFLAGS) $(CFLAGSNO) -o $@ $@.o `pkg-config --libs opencv4` $(LIBSNO)

#main: $(OBJS)
#	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

main: $(OBJS)
	$(CC) $(LDFLAGS) $(CFLAGS) $(OBJS) -o $@ `pkg-config --libs opencv4` $(LIBS)
.cpp.o:
	$(CC) $(CFLAGS) -c $<
