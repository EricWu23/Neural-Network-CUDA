# Add your Source files to this variable
SOURCESLOCATION =linear.cpp \
      main.cpp \
      mse.cpp \
      relu.cpp \
      sequential.cpp \
      train.cpp \
      $(UTILS)/utils.cpp \
      $(DATA)/read_csv.cpp
# Add your Source files to this variable
SOURCES = linear.cpp \
      main.cpp \
      mse.cpp \
      relu.cpp \
      sequential.cpp \
      train.cpp \
      utils.cpp \
      read_csv.cpp
# Add your files to be cleaned but not part of the project
IRRELEVANT =

# Add your include paths to this variable
UTILS:=../utils
DATA:=../data

INCLUDES = $(UTILS) \
           $(DATA)
