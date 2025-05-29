#include "convert.hpp"
#include "OBJToBNDConverter.hpp"

int main(){
	
	Convert("../data/initial_struct.bnd");
	ObjToBndConverter converter;
	if (!converter.loadObjFile("initial_struct.obj")) {
        std::cerr << "Failed to load OBJ file." << std::endl;
        return 1;
    }
    
    if (!converter.writeBndFile("recover.bnd")) {
        std::cerr << "Failed to write BND file." << std::endl;
        return 1;
    }
	Convert("recover.bnd");
	return 1;
}