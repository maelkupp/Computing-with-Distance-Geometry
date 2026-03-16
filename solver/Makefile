# ====== CONFIGURATION ======
GUROBI_HOME = /opt/gurobi1100/linux64

CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -I$(GUROBI_HOME)/include
LDFLAGS = -L$(GUROBI_HOME)/lib -lgurobi_c++ -lgurobi110

# ====== FILES ======
SRC = main.cpp dgp_bnb.cpp
OBJ = $(SRC:.cpp=.o)

TARGET = solver

# ====== RULES ======

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
