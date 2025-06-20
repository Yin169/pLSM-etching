#include "convert.hpp"
// #include "alphawrap.hpp"
#include "OBJToBNDConverter.hpp"

int main(){
	
	// Convert("../data/Silicon_etch_result.bnd");
    // Wrapper("./initial_struct.obj", 600, 600);

    std::string obj_file = "../out/Silicon_etch.obj";
    std::string bnd_file = "Silicon_etch_allregion.bnd";
    
    if (ConvertOBJToDFISE(obj_file, bnd_file)) {
        std::cerr << "Conversion failed!" << std::endl;
        return 1;
    }

	return 1;
}