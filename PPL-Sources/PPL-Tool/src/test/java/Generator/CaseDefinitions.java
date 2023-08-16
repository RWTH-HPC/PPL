package Generator;

import java.util.HashMap;

/**
 * Enum describing all test cases for a parameterized test.
 * The Cases are encoded in the following way:
 *
 * (P/N/G/C)_(N/C/A/AC/MA/MC/MAC/AA)_Name
 *
 * The shorthands stand for the following directories in the first part
 * P: Plain
 * N: Nested
 * G: General
 * C: Concatenated
 *
 * The shorthands in the second part mean the following directories
 * N: None
 * C: CPU
 * A: Accelerator
 * AC: Accelerator-CPU
 * MA: MPI-Accelerator
 * MC: MPI-CPU
 * MAC: MPI-Accelerator-CPU
 * AA: Multiple-Accelerator
 */
public enum CaseDefinitions {

    S_DM_G,
    S_DM_L,
    S_DM_LWI,
    S_DM_M,

    S_I_DI,
    S_I_DNI,

    S_S_CS_FL,
    S_S_CS_FEL,
    S_S_CS_B,
    S_S_CS_WL,

    S_S_E_ARITH,
    S_S_E_ARRAY,
    S_S_E_BOOL,
    S_S_E_VAR,


    P_C_B_C_AC_DP,
    P_C_B_C_AC_M,
    P_C_B_C_AC_S,
    P_C_B_C_AC_RED,
    P_C_B_C_AC_REC,

    P_C_B_C_MA_DP,
    P_C_B_C_MA_M,
    P_C_B_C_MA_S,
    P_C_B_C_MA_RED,
    P_C_B_C_MA_REC,

    P_C_B_C_MAC_DP,
    P_C_B_C_MAC_M,
    P_C_B_C_MAC_S,
    P_C_B_C_MAC_RED,
    P_C_B_C_MAC_REC,

    P_C_B_C_MC_DP,
    P_C_B_C_MC_M,
    P_C_B_C_MC_S,
    P_C_B_C_MC_RED,
    P_C_B_C_MC_REC,

    P_C_B_C_AA_DP,
    P_C_B_C_AA_M,
    P_C_B_C_AA_S,
    P_C_B_C_AA_RED,
    P_C_B_C_AA_REC,

    P_C_B_M_ASA,
    P_C_B_M_ASC,
    P_C_B_M_ACCESS,
    P_C_B_M_CSA,
    P_C_B_M_CSC,
    P_C_B_M_FLOWAC,
    P_C_B_M_FLOWCA,
    P_C_B_M_FLOWNN,
    P_C_B_M_FLOWOVERLAP,
    P_C_B_M_PAR,
    P_C_B_M_PIPEA,
    P_C_B_M_PIPEC,
    P_C_B_M_READCA,
    P_C_B_M_READNN,
    P_C_B_M_READOVERLAP,
    P_C_B_M_WRITEOVERLAP,

    P_C_O_INTER_A_DP,
    P_C_O_INTER_A_M,
    P_C_O_INTER_A_S,
    P_C_O_INTER_A_RED,
    P_C_O_INTER_A_REC,

    P_C_O_INTER_C_DP,
    P_C_O_INTER_C_M,
    P_C_O_INTER_C_S,
    P_C_O_INTER_C_RED,
    P_C_O_INTER_C_REC,

    P_C_O_INTRA_A_DP,
    P_C_O_INTRA_A_M,
    P_C_O_INTRA_A_S,
    P_C_O_INTRA_A_RED,
    P_C_O_INTRA_A_REC,

    P_C_O_INTRA_C_DP,
    P_C_O_INTRA_C_M,
    P_C_O_INTRA_C_S,
    P_C_O_INTRA_C_RED,
    P_C_O_INTRA_C_REC,

    P_C_O_INTRA_A_DPN,
    P_C_O_INTRA_A_MN,
    P_C_O_INTRA_A_SN,
    P_C_O_INTRA_A_REDN,
    P_C_O_INTRA_A_RECN,

    P_C_O_INTRA_C_DPN,
    P_C_O_INTRA_C_MN,
    P_C_O_INTRA_C_SN,
    P_C_O_INTRA_C_REDN,
    P_C_O_INTRA_C_RECN,
    ;


    private static final String globalPath = "../../TestSuite/Verification/Error/";
    private static final String serial = "Serial/";
    private static final String parallel = "Parallel/";

    private static final String data_management = "Data-Management/";
    private static final String inlining = "Inlining/";
    private static final String syntax = "Syntax/";

    private static final String expression = "Expression/";
    private static final String control_Statement = "Control-Statement/";

    private static final String correctness = "Correctness/";

    private static final String between_Multiple_Devices = "Between-Multiple-Devices/";
    private static final String on_Device = "On-Device/";

    private static final String concurrency_Defect = "Concurrency-Defect/";
    private static final String mapping_Defect = "Mapping-Defect/";

    private static final String inter = "Inter-Region/";
    private static final String intra = "Intra-Region/";


    private static final String CPUTests = "CPU/";
    private static final String ACCTests = "Accelerator/";
    private static final String ACC_CPUTests = "Accelerator-CPU/";
    private static final String MPI_ACCTests = "MPI-Accelerator/";
    private static final String MPI_CPUTests = "MPI-CPU/";
    private static final String MPI_ACC_CPUTests = "MPI-CPU-Accelerator/";
    private static final String Multi_ACCTests = "Multiple-Accelerator/";

    public static final HashMap<CaseDefinitions, TestCase> paths;
    static {
        paths = new HashMap<>();
        paths.put(S_DM_G, new TestCase(globalPath + serial + data_management, "GlobalAllocationTest"));
        paths.put(S_DM_L, new TestCase(globalPath + serial + data_management, "LocalAllocationTest"));
        paths.put(S_DM_LWI, new TestCase(globalPath + serial + data_management, "LocalAllocationTestWithInlining"));
        paths.put(S_DM_M, new TestCase(globalPath + serial + data_management, "MainAllocationTest"));

        paths.put(S_I_DI, new TestCase(globalPath + serial + inlining, "DoInlineTest"));
        paths.put(S_I_DNI, new TestCase(globalPath + serial + inlining, "DoNotInlineTest"));

        paths.put(S_S_CS_B, new TestCase(globalPath + serial + syntax + control_Statement, "BranchTest"));
        paths.put(S_S_CS_FEL, new TestCase(globalPath + serial + syntax + control_Statement, "ForEachLoopTest"));
        paths.put(S_S_CS_WL, new TestCase(globalPath + serial + syntax + control_Statement, "WhileLoopTest"));
        paths.put(S_S_CS_FL, new TestCase(globalPath + serial + syntax + control_Statement, "ForLoopTest"));

        paths.put(S_S_E_ARITH, new TestCase(globalPath + serial + syntax + expression, "ArithmeticExpressionTest"));
        paths.put(S_S_E_ARRAY, new TestCase(globalPath + serial + syntax + expression, "ArrayExpressionTest"));
        paths.put(S_S_E_BOOL, new TestCase(globalPath + serial + syntax + expression, "BooleanExpressionTest"));
        paths.put(S_S_E_VAR, new TestCase(globalPath + serial + syntax + expression, "VariableGenerationTest"));


        paths.put(P_C_B_C_AC_DP, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + ACC_CPUTests, "DPTest"));
        paths.put(P_C_B_C_AC_M, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + ACC_CPUTests, "MapTest"));
        paths.put(P_C_B_C_AC_S, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + ACC_CPUTests, "StencilTest"));
        paths.put(P_C_B_C_AC_RED, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + ACC_CPUTests, "ReduceTest"));
        paths.put(P_C_B_C_AC_REC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + ACC_CPUTests, "RecursionTest"));

        paths.put(P_C_B_C_MA_DP, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACCTests, "DPTest"));
        paths.put(P_C_B_C_MA_M, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACCTests, "MapTest"));
        paths.put(P_C_B_C_MA_S, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACCTests, "StencilTest"));
        paths.put(P_C_B_C_MA_RED, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACCTests, "ReduceTest"));
        paths.put(P_C_B_C_MA_REC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACCTests, "RecursionTest"));

        paths.put(P_C_B_C_MAC_DP, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACC_CPUTests, "DPTest"));
        paths.put(P_C_B_C_MAC_M, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACC_CPUTests, "MapTest"));
        paths.put(P_C_B_C_MAC_S, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACC_CPUTests, "StencilTest"));
        paths.put(P_C_B_C_MAC_RED, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACC_CPUTests, "ReduceTest"));
        paths.put(P_C_B_C_MAC_REC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_ACC_CPUTests, "RecursionTest"));

        paths.put(P_C_B_C_MC_DP, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_CPUTests, "DPTest"));
        paths.put(P_C_B_C_MC_M, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_CPUTests, "MapTest"));
        paths.put(P_C_B_C_MC_S, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_CPUTests, "StencilTest"));
        paths.put(P_C_B_C_MC_RED, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_CPUTests, "ReduceTest"));
        paths.put(P_C_B_C_MC_REC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + MPI_CPUTests, "RecursionTest"));

        paths.put(P_C_B_C_AA_DP, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + Multi_ACCTests, "DPTest"));
        paths.put(P_C_B_C_AA_M, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + Multi_ACCTests, "MapTest"));
        paths.put(P_C_B_C_AA_S, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + Multi_ACCTests, "StencilTest"));
        paths.put(P_C_B_C_AA_RED, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + Multi_ACCTests, "ReduceTest"));
        paths.put(P_C_B_C_AA_REC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + concurrency_Defect + Multi_ACCTests, "RecursionTest"));

        paths.put(P_C_B_M_ASA, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "ACC2SEQ2ACC"));
        paths.put(P_C_B_M_ASC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "ACC2SEQ2CPU"));
        paths.put(P_C_B_M_ACCESS, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "ChangedAccessScheme"));
        paths.put(P_C_B_M_CSA, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "CPU2SEQ2ACC"));
        paths.put(P_C_B_M_CSC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "CPU2SEQ2CPU"));
        paths.put(P_C_B_M_FLOWAC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "FlowACC2CPU"));
        paths.put(P_C_B_M_FLOWCA, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "FlowCPU2ACC"));
        paths.put(P_C_B_M_FLOWNN, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "FlowNODE2NODE"));
        paths.put(P_C_B_M_FLOWOVERLAP, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect,"FlowPartialOverlap"));
        paths.put(P_C_B_M_PAR, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect,"ParallelExecution"));
        paths.put(P_C_B_M_PIPEA, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "PipelineACC"));
        paths.put(P_C_B_M_PIPEC, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "PipelineCPU"));
        paths.put(P_C_B_M_READCA, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect, "ReadCPU2ACC"));
        paths.put(P_C_B_M_READNN, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect,"ReadNODE2NODE"));
        paths.put(P_C_B_M_READOVERLAP, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect,"ReadPartialOverlap"));
        paths.put(P_C_B_M_WRITEOVERLAP, new TestCase(globalPath + parallel + correctness + between_Multiple_Devices + mapping_Defect,"WritePartialOverlap"));

        paths.put(P_C_O_INTER_A_DP, new TestCase(globalPath + parallel + correctness + on_Device + inter + ACCTests, "DPTest"));
        paths.put(P_C_O_INTER_A_M, new TestCase(globalPath + parallel + correctness + on_Device + inter + ACCTests, "MapTest"));
        paths.put(P_C_O_INTER_A_S, new TestCase(globalPath + parallel + correctness + on_Device + inter + ACCTests, "StencilTest"));
        paths.put(P_C_O_INTER_A_RED, new TestCase(globalPath + parallel + correctness + on_Device + inter + ACCTests, "ReduceTest"));
        paths.put(P_C_O_INTER_A_REC, new TestCase(globalPath + parallel + correctness + on_Device + inter + ACCTests, "RecursionTest"));

        paths.put(P_C_O_INTER_C_DP, new TestCase(globalPath + parallel + correctness + on_Device + inter + CPUTests, "DPTest"));
        paths.put(P_C_O_INTER_C_M, new TestCase(globalPath + parallel + correctness + on_Device + inter + CPUTests, "MapTest"));
        paths.put(P_C_O_INTER_C_S, new TestCase(globalPath + parallel + correctness + on_Device + inter + CPUTests, "StencilTest"));
        paths.put(P_C_O_INTER_C_RED, new TestCase(globalPath + parallel + correctness + on_Device + inter + CPUTests, "ReduceTest"));
        paths.put(P_C_O_INTER_C_REC, new TestCase(globalPath + parallel + correctness + on_Device + inter + CPUTests, "RecursionTest"));

        paths.put(P_C_O_INTRA_A_DP, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "DPTest"));
        paths.put(P_C_O_INTRA_A_M, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "MapTest"));
        paths.put(P_C_O_INTRA_A_S, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "StencilTest"));
        paths.put(P_C_O_INTRA_A_RED, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "ReduceTest"));
        paths.put(P_C_O_INTRA_A_REC, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "RecursionTest"));

        paths.put(P_C_O_INTRA_C_DP, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "DPTest"));
        paths.put(P_C_O_INTRA_C_M, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "MapTest"));
        paths.put(P_C_O_INTRA_C_S, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "StencilTest"));
        paths.put(P_C_O_INTRA_C_RED, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "ReduceTest"));
        paths.put(P_C_O_INTRA_C_REC, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "RecursionTest"));

        paths.put(P_C_O_INTRA_A_DPN, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "DPNestedTest"));
        paths.put(P_C_O_INTRA_A_MN, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "MapNestedTest"));
        paths.put(P_C_O_INTRA_A_SN, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "StencilNestedTest"));
        paths.put(P_C_O_INTRA_A_REDN, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "ReduceNestedTest"));
        paths.put(P_C_O_INTRA_A_RECN, new TestCase(globalPath + parallel + correctness + on_Device + intra + ACCTests, "RecursionNestedTest"));

        paths.put(P_C_O_INTRA_C_DPN, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "DPNestedTest"));
        paths.put(P_C_O_INTRA_C_MN, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "MapNestedTest"));
        paths.put(P_C_O_INTRA_C_SN, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "StencilNestedTest"));
        paths.put(P_C_O_INTRA_C_REDN, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "ReduceNestedTest"));
        paths.put(P_C_O_INTRA_C_RECN, new TestCase(globalPath + parallel + correctness + on_Device + intra + CPUTests, "RecursionNestedTest"));
    }


}
