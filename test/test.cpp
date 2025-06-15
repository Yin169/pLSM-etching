#include "convert.hpp"
#include "alphawrap.hpp"
#include "OBJToBNDConverter.hpp"

int main(){
	
	Convert("./data/Silicon_etch_result.bnd");
    Wrapper("./initial_struct.obj", 600, 600);

    std::string obj_file = "./initial_struct_600_600.obj";
    std::string bnd_file = "./recover.bnd";
    std::string material = "Polymer";
    std::string region = "Polymer_1";

    ObjToDfiseConverter converter(material, region);
    
    if (!converter.convert(obj_file, bnd_file)) {
        std::cerr << "Conversion failed!" << std::endl;
        return 1;
    }

	return 1;
}