CXX             = g++ 
SRCS            = $(wildcard *.cpp) $(wildcard *.c)
OBJS            = $(SRCS:.cpp=.o)
TARGET          = out 
LIBS            = -lmvrlu-ordo -lpthread
LIB_DIRS        = -L./lib
INC             = -I./include
 
all : $(TARGET)
		$(CXX) -o $(TARGET) $(OBJS) $(INC) $(LIB_DIRS) $(LIBS)
		rm -f *.o

$(TARGET) :
		$(CXX) -c $(SRCS) $(INC) $(LIB_DIRS) $(LIBS)

clean :
		rm -f *.o
		rm -f $(TARGET)