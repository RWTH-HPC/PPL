
/*********************************************************************/
/*        This is a generated C++ Header file.                       */
/*        Generated by PatternDSL.                                   */
/*        Contains the function declarations for the source file     */
/*********************************************************************/

#ifndef heartwall_g_1_HXX
#define heartwall_g_1_HXX

uint randomizer(int n = 42) {
    srand(n);
    return rand();
}

    int32_t* kernel_func(int32_t* exchange_data, int32_t frame_no, float* frame, int32_t public_mask_conv_ioffset, int32_t public_mask_conv_joffset);


#endif