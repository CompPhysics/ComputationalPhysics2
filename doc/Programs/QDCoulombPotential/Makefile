# http://mrbook.org/blog/tutorials/make/
# http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/

CC=g++
CFLAGS= -std=c++0x -Wall -g -o3 #-Wextra -pedantic -std=c++11
CXXFLAGS=
LIBS= -lm
EXECUTABLES = QD_Coulomb

O_FILES = $(patsubst %.cpp, %.o, $(wildcard *.cpp))

$(EXECUTABLES): $(O_FILES)
	$(CC) $(CXXFLAGS) $(CFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp  
	$(CC) $(CXXFLAGS) $(CFLAGS) -c -o $@ $< 

clean:
	rm *.o *~ *# 
