GXX=/usr/bin/g++
GXXFLAGS=-std=c++17 -O3 -Wall -fmessage-length=0
GXXOMPFLAG=-fopenmp

all: action.o agent.o board.o episode.o statistics.o
	$(GXX) $(GXXFLAGS) $(GXXOMPFLAG) -o threes threes.cpp
action.o: action.h
	$(GXX) $(GXXFLAGS) -c action.h
agent.o: agent.h action.o board.o weight.o
	$(GXX) $(GXXFLAGS) -c agent.h
board.o: board.h
	$(GXX) $(GXXFLAGS) -c board.h
episode.o: episode.h action.o agent.o board.o
	$(GXX) $(GXXFLAGS) -c episode.h
statistics.o: statistics.h action.o board.o episode.o
	$(GXX) $(GXXFLAGS) -c statistics.h
weight.o: weight.h
	$(GXX) $(GXXFLAGS) -c weight.h
stats:
	./threes --total=1000 --save=stats.txt
clean:
	rm threes
	rm *.h.gch
