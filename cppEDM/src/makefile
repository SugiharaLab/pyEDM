
.PHONY: all clean distclean depend 

CXX = g++

HEADERS = API.h CCM.h Common.h DataFrame.h DateTime.h EDM.h EDM_Neighbors.h\
          Multiview.h Parameter.h Simplex.h SMap.h Version.h

SRCS = API.cc CCM.cc Common.cc DateTime.cc EDM.cc EDM_Formatting.cc\
       EDM_Neighbors.cc Eval.cc Multiview.cc Parameter.cc Simplex.cc SMap.cc

OBJ = $(SRCS:%.cc=%.o)

LIB = libEDM.a

CFLAGS += -std=c++11 -Wpedantic -Wall -Wextra -Wreorder -O3
CFLAGS += -DCCM_THREADED -fPIC
# CFLAGS += -g # -DDEBUG_ALL
# LFLAGS = -L./ -lstdc++ -lEDM -lpthread -llapacke -llapack -lblas 

all:	$(LIB)
	cp $(LIB) ../lib/

clean:
	rm -f $(OBJ) $(LIB)

distclean:
	rm -f $(OBJ) $(LIB) ../lib/$(LIB) *~ *.bak *.csv

$(LIB): $(OBJ)
	ar -rcs $(LIB) $(OBJ)

%.o : %.cc 
	$(CXX) $(CFLAGS) -c $<

depend:
	@echo ${SRCS}
	makedepend -Y $(SRCS)
# DO NOT DELETE

API.o: API.h Common.h DataFrame.h Parameter.h Version.h Simplex.h EDM.h
API.o: SMap.h CCM.h Multiview.h
CCM.o: CCM.h EDM.h Common.h DataFrame.h Parameter.h Version.h Simplex.h
Common.o: Common.h DataFrame.h
DateTime.o: DateTime.h
EDM.o: EDM.h Common.h DataFrame.h Parameter.h Version.h
EDM_Formatting.o: EDM.h Common.h DataFrame.h Parameter.h Version.h DateTime.h
EDM_Neighbors.o: EDM_Neighbors.h EDM.h Common.h DataFrame.h Parameter.h
EDM_Neighbors.o: Version.h
Eval.o: API.h Common.h DataFrame.h Parameter.h Version.h Simplex.h EDM.h
Eval.o: SMap.h CCM.h Multiview.h
Multiview.o: Multiview.h EDM.h Common.h DataFrame.h Parameter.h Version.h
Multiview.o: Simplex.h
Parameter.o: Parameter.h Common.h DataFrame.h Version.h
Simplex.o: Simplex.h EDM.h Common.h DataFrame.h Parameter.h Version.h
SMap.o: SMap.h EDM.h Common.h DataFrame.h Parameter.h Version.h
