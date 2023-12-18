
/*********************************************************************/
/*        This is a generated C++ Header file.                       */
/*        Generated by PatternDSL.                                   */
/*        Contains the function declarations for the source file     */
/*********************************************************************/

#ifndef a_HXX
#define a_HXX

    double* luleshSeq_calcHourgam(double* xyz8n, double* dvd, double* gamma, double volinv);
    int32_t luleshSeq_checkNonPosElem(double* array);
    int32_t checkGreaterThreshold(double* array, double threshold);
    double AreaFace(double x0, double x1, double x2, double x3, double y0, double y1, double y2, double y3, double z0, double z1, double z2, double z3);
    double luleshSeq_CalcElemCharacteristicLength(double* x, double* y, double* z, double volume);
    double* luleshSeq_replaceDomainNodelist(double* inputArray, int32_t index0, int32_t index1, int32_t index2, int32_t dim);
    double* CalcMonotonicQForElems(double* domain_coordGradients, double* domain_velocGradients, double* domain_vdov, double* domain_elemMass, double* domain_volo, double* domain_vnew);
    double* luleshSeq_CalcElemVelocityGradient(double* xvel, double* yvel, double* zvel, double* B, double detJ);
    double* luleshSeq_CalcElemNodeNormals(double* B, double* coordinates_local);
    double* luleshSeq_ApplyAccelerationBoundaryConditionsForNodes_check(double* domain_accelerations_old_1d, int32_t idx0, int32_t idx1, int32_t idx2);
    double luleshSeq_CalcHydroConstraintForElems(double* domain_vdov, double dthydro);
    double* luleshSeq_CalcElemFBHourglassForce(double* d, double* hourgam, double coefficient);
    double luleshSeq_getWishedValue1D(double* array, int32_t wishedIndex);
    double* luleshSeq_ApplyMaterialPropertiesForElems(double* domain_vnew, double* domain_v, double* domain_e, double* domain_delv, double* domain_p, double* domain_q, double* domain_qq, double* domain_ql);
    int32_t luleshSeq_checkNonPosElem_qGcPoyNonj(double* array);
    double* VoluDer(double x0, double x1, double x2, double x3, double x4, double x5, double y0, double y1, double y2, double y3, double y4, double y5, double z0, double z1, double z2, double z3, double z4, double z5);
    double* luleshSeq_CalcPressureForElems(double vnewc, double compression, double e_old);
    double* luleshSeq_CalcQForElems(double* domain_q, double* domain_volo, double* domain_vnew, double* domain_coordinates, double* domain_velocities, double* domain_vdov, double* domain_elemMass);
    int32_t lulesh_util_VerifyAndWriteFinalOutput(double elapsed_time, int64_t used_peak_memory, int32_t nx, int32_t domain_cycle, double* domain_e);
    double luleshSeq_CalcSoundSpeedForElems(double pbvc, double bvc, double enewc, double pnewc, double vnewc);
    double* luleshSeq_getWishedValue2D(double* array, int32_t wishedIndex);
    double luleshSeq_CalcElemVolume(double* x, double* y, double* z);
    double* luleshSeq_CollectDomainNodesToElemNodes(double* inputArray, int32_t index0, int32_t index1, int32_t index2);
    double* luleshSeq_CalcElemShapeFunctionDerivatives(double* coordinates_local);
    double luleshSeq_get3DSum(double* array);
    double* luleshSeq_CalcElemVolumeDerivative(double* xyz);
    double* SumElemFaceNormal(double* B, double* c_l, int32_t index0, int32_t index1, int32_t index2, int32_t index3);
    double* luleshSeq_CalcFBHourglassForceForElems(double* domain_ss, double* domain_elemMass, double* domain_velocities, double* domain_forces, double* determ, double* xyz8n, double* dvd, double hourg);
    double luleshSeq_CalcCourantConstraintForElems(double qqc, double dtcourant, double* domain_ss, double* domain_arealg, double* domain_vdov);
    int32_t luleshSeq_checkNonPosElem_aOsFcfJMDl(double* array);
    int32_t luleshSeq_checkNonPosElem2(double* array);
    double* luleshSeq_SumElemStressesToNodeForces(double* sigXYZ, double* B);
    double TRIPLE_PRODUCT(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3);
    double* getLocalElement(double* array, int32_t index);


#endif