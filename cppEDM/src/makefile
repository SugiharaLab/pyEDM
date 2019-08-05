
CC  = g++
OBJ = Common.o AuxFunc.o DateTimeUtil.o Parameter.o Embed.o Interface.o\
	Neighbors.o Simplex.o Eval.o CCM.o Multiview.o SMap.o 

LIB = libEDM.a

CFLAGS = -std=c++11 -I../../ -fPIC -DCCM_THREADED -DMULTIVIEW_VALUES_OVERLOAD -O3 # -g -DDEBUG -DDEBUG_ALL
LFLAGS = -L./ -lstdc++  -lEDM -lpthread # -llapacke -llapack -lblas 

all:	$(LIB)
	ar -rcs $(LIB) $(OBJ)
	cp $(LIB) ../lib/

clean:
	rm -f $(OBJ) $(LIB)

distclean:
	rm -f $(OBJ) $(LIB) ../lib/$(LIB) *~ *.bak *.csv

$(LIB): $(OBJ)

AuxFunc.o: AuxFunc.cc
	$(CC) -c AuxFunc.cc $(CFLAGS)

DateTimeUtil.o: DateTimeUtil.cc
	$(CC) -c DateTimeUtil.cc $(CFLAGS)

Common.o: Common.cc
	$(CC) -c Common.cc $(CFLAGS)

Parameter.o: Parameter.cc
	$(CC) -c Parameter.cc $(CFLAGS)

Embed.o: Embed.cc
	$(CC) -c Embed.cc $(CFLAGS)

Interface.o: Interface.cc
	$(CC) -c Interface.cc $(CFLAGS)

Neighbors.o: Neighbors.cc
	$(CC) -c Neighbors.cc $(CFLAGS)

Simplex.o: Simplex.cc
	$(CC) -c Simplex.cc $(CFLAGS)

Eval.o: Eval.cc
	$(CC) -c Eval.cc $(CFLAGS)

CCM.o: CCM.cc
	$(CC) -c CCM.cc $(CFLAGS)

Multiview.o: Multiview.cc
	$(CC) -c Multiview.cc $(CFLAGS)

SMap.o: SMap.cc
	$(CC) -c SMap.cc $(CFLAGS)

SRCS = `echo ${OBJ} | sed -e 's/.o /.cc /g'`
depend:
	@echo ${SRCS}
	makedepend -Y $(SRCS)
# DO NOT DELETE

Common.o: Common.h DataFrame.h
AuxFunc.o: AuxFunc.h Common.h DataFrame.h Neighbors.h Parameter.h Version.h
AuxFunc.o: Embed.h DateTime.h
DateTimeUtil.o: DateTime.h
Parameter.o: Parameter.h Common.h DataFrame.h Version.h
Embed.o: Embed.h Common.h DataFrame.h Parameter.h Version.h
Interface.o: Common.h DataFrame.h
Neighbors.o: Neighbors.h Common.h DataFrame.h Parameter.h Version.h
Simplex.o: Common.h DataFrame.h Parameter.h Version.h Neighbors.h Embed.h
Simplex.o: AuxFunc.h
Eval.o: Common.h DataFrame.h
CCM.o: Common.h DataFrame.h Embed.h Parameter.h Version.h AuxFunc.h
CCM.o: Neighbors.h
Multiview.o: Common.h DataFrame.h AuxFunc.h Neighbors.h Parameter.h Version.h
Multiview.o: Embed.h
