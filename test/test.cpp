#include "convert.hpp"
#include "OBJToBNDConverter.hpp"

int main(){
	
	Convert("../data/Silicon_etch_result.bnd");
	ObjToBndConverter converter;
	if (!converter.loadObjFile("initial_struct.obj")) {
        std::cerr << "Failed to load OBJ file." << std::endl;
        return 1;
    }
    
    if (!converter.writeBndFile("recover.bnd")) {
        std::cerr << "Failed to write BND file." << std::endl;
        return 1;
    }
	return 1;
}