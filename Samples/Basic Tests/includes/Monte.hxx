
/*********************************************************************/
/*        This is a generated C++ Header file.                       */
/*        Generated by PatternDSL.                                   */
/*        Contains the function declarations for the source file     */
/*********************************************************************/

#ifndef Monte_HXX
#define Monte_HXX

uint randomizer(int n = 42) {
    srand(n);
    return rand();
}

    float uniform(int32_t x);
    int32_t lehmer_random_number_generator(int32_t x);


#endif