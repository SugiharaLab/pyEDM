#ifndef VERSION_H
#define VERSION_H

//------------------------------------------------------------
// Instantiated in Parameters() constructor
//------------------------------------------------------------
class Version {
    int         Major;
    int         Minor;
    int         Micro;
    std::string Date;
    
public:
    Version( int Major, int Minor, int Micro, std::string Date ) :
        Major( Major  ), Minor( Minor ), Micro( Micro ), Date ( Date  ) {};
    
    void ShowVersion() {
        std::cout << "cppEDM Version " << Major << "."
                  << Minor << "." << Micro << " " << Date << std::endl;
    }
};
#endif
