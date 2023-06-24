# specify the C++ compiler to use
CXX = g++

CXXFLAGS = -g -Wall

# specify the linker flags
LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system

# the name of the executable
EXE = sfml-app

# the object files
OBJS = main.o

$(EXE): $(OBJS)
	$(CXX) $(OBJS) -o $(EXE) $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

clean:
	rm -f $(EXE) $(OBJS)

