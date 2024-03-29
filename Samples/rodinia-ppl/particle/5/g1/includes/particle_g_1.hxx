
/*********************************************************************/
/*        This is a generated C++ Header file.                       */
/*        Generated by PatternDSL.                                   */
/*        Contains the function declarations for the source file     */
/*********************************************************************/

#ifndef particle_g_1_HXX
#define particle_g_1_HXX

uint randomizer(int n = 42) {
    srand(n);
    return rand();
}

    double randu(int32_t* seed, int32_t index);
    int32_t* imdilate_disk(int32_t* matrix, int32_t dimX, int32_t dimY, int32_t dimZ, int32_t error);
    int32_t* videoSequence(int32_t* seed);
    double single_particle(double* array, double* CDF, int32_t Nparticles, double u);
    int32_t* setIf(int32_t testValue, int32_t newValue, int32_t* array3D, int32_t dimX, int32_t dimY, int32_t dimZ);
    double randn(int32_t* seed, int32_t index);
    int32_t* addNoise(int32_t* array3D, int32_t dimX, int32_t dimY, int32_t dimZ, int32_t* seed);
    int32_t* dilate_matrix(int32_t* matrix, int32_t posX, int32_t posY, int32_t posZ, int32_t dimX, int32_t dimY, int32_t dimZ, int32_t error);
    float powi(float x, int32_t n);
    int32_t* strelDisk(int32_t radius);
    int32_t findIndex(double* CDF, int32_t lengthCDF, double value);
    double partialLikelihood(int32_t* I, int32_t k, double arrayX, double arrayY, double* objxy, int32_t countOnes);
    double* getneighbors(int32_t* se, int32_t numOnes, int32_t radius);
    int32_t roundDouble(double value);


#endif