package de.parallelpatterndsl.patterndsl.generator;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.RecursionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.SerialMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.RecursionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.printer.CppExpressionPrinter;
import de.parallelpatterndsl.patterndsl.printer.CppPlainNodePrinter;
import de.parallelpatterndsl.patterndsl.printer.CppTypesPrinter;
import org.javatuples.Quartet;
import org.javatuples.Triplet;

import java.util.ArrayList;
// TODO: clean up and rework
/**
 * A number of functions used within the templates.
 */
public class PatternDSLGeneratorHelper {

    private AbstractMappingTree AMT;

    private String filename;

    private Network network;
    private String gpuThread;

    /**
     * contains the generated sources:
     * value0: Main source file.
     * value1: GPU/CUDA source file.
     * value2: GPU/CUDA header file.
     * value3: GPU/CUDA kernel header file.
     */
    private Quartet<String, String, String, String> generatedSource;

    public PatternDSLGeneratorHelper(AbstractMappingTree AMT, String filename, Network network, String gpuThread) {
        this.AMT = AMT;
        this.filename = filename;
        this.network = network;
        this.gpuThread = gpuThread;
    }

    public String getFilename() {
        return filename;
    }

    public String GPUpinning() {
        return gpuThread;
    }

    /**
     * Returns the contents of the machine file
     * @return
     */
    public String getMachines() {
        StringBuilder bob = new StringBuilder();
        for (Node node: network.getNodes()) {
            bob.append(node.getAddress());
            bob.append("\n");
        }

        return bob.toString();
    }

    /**
     * Returns a list containing all serial function
     * @return
     */
    public ArrayList<SerialMapping> getSerialFunctions() {
        ArrayList<SerialMapping> result = new ArrayList<>();

        for ( FunctionMapping func: AbstractMappingTree.getFunctionTable().values()) {
            if (func instanceof SerialMapping) {
                result.add((SerialMapping) func);
            }
        }
        return result;
    }

    /**
     * Returns a list containing all recursive functions
     * @return
     */
    public ArrayList<RecursionMapping> getRecursiveFunctions() {
        ArrayList<RecursionMapping> result = new ArrayList<>();

        for ( FunctionMapping func: AbstractMappingTree.getFunctionTable().values()) {
            if (func instanceof RecursionMapping) {
                result.add((RecursionMapping) func);
            }
        }
        return result;
    }

    /**
     * prints the function type of a given function
     * @param function
     * @return
     */
    public String printFunctionType(FunctionMapping function) {
        return CppPlainNodePrinter.printFunctionType(function);
    }

    /**
     * prints the arguments of a given function
     * @param function
     * @return
     */
    public String printArguments(FunctionMapping function) {
        return CppPlainNodePrinter.printArguments(function);
    }

    /**
     * Prints the root node with the appropriate visitor.
     * @return
     */
    public String printRoot() {
        if (generatedSource == null) {
            generatedSource = CppPlainNodePrinter.doPrintNode(AMT.getRoot(), AMT, network);
        }
        return generatedSource.getValue0();
    }

    /**
     * Prints the cuda code with the appropriate visitor.
     * @return
     */
    public String printCuda() {
        if (generatedSource == null) {
            generatedSource = CppPlainNodePrinter.doPrintNode(AMT.getRoot(), AMT, network);
        }
        return generatedSource.getValue1();
    }

    /**
     * Prints the cuda header code with the appropriate visitor.
     * @return
     */
    public String printCudaHeader() {
        if (generatedSource == null) {
            generatedSource = CppPlainNodePrinter.doPrintNode(AMT.getRoot(), AMT, network);
        }
        return generatedSource.getValue2();
    }

    /**
     * Prints the cuda kernel header code with the appropriate visitor.
     * @return
     */
    public String printCudaKernelHeader() {
        if (generatedSource == null) {
            generatedSource = CppPlainNodePrinter.doPrintNode(AMT.getRoot(), AMT, network);
        }
        return generatedSource.getValue3();
    }

    /**
     * Prints the declaration of the global variables.
     * @return
     */
    public String printGlobalVars(boolean isHeader) {
        CppExpressionPrinter.setAMT(AMT);
        StringBuilder builder = new StringBuilder();
        for (Data data: AMT.getGlobalVariableTable().values() ) {
            if (data instanceof PrimitiveData) {
                continue;
            }
            if (!( data instanceof FunctionInlineData)) {
                boolean doGlobalInit = false;
                if (data instanceof ArrayData) {
                    if (!((ArrayData) data).isOnStack()) {
                        doGlobalInit = true;
                    }
                } else {
                    doGlobalInit = true;
                }
                if (doGlobalInit) {
                    if (isHeader) {
                        builder.append("extern ");
                    }
                    builder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                    builder.append(" ");
                    if (data instanceof ArrayData) {
                        builder.append("*");
                    }
                    builder.append(data.getIdentifier());
                    builder.append(";\n");
                }
            }
        }

        if (!isHeader) {
            for (IRLExpression exp : AMT.getGlobalAssignments()) {
                if (exp instanceof AssignmentExpression) {
                    if (((AssignmentExpression) exp).getOutputElement() instanceof ArrayData) {
                        if (((ArrayData) ((AssignmentExpression) exp).getOutputElement()).isOnStack()) {
                            builder.append(CppExpressionPrinter.doPrintExpression(exp, false, new ArrayList<>(), new ArrayList<>()));
                            builder.append(";\n");
                        }
                    }
                }
            }
        }

        if (isHeader) {
            ArrayList<IRLExpression> newGlobalAssignments = new ArrayList<>();
            // Generate constant arrays
            for (IRLExpression exp : AMT.getGlobalAssignments()) {
                if (exp instanceof AssignmentExpression) {
                    if (((AssignmentExpression) exp).getOutputElement() instanceof ArrayData) {
                        if (((ArrayData) ((AssignmentExpression) exp).getOutputElement()).isOnStack()) {
                            //builder.append(CppExpressionPrinter.doPrintExpression(exp, false, new ArrayList<>(), new ArrayList<>()));
                            builder.append("extern ");
                            builder.append((CppTypesPrinter.doPrintType(((AssignmentExpression) exp).getOutputElement().getTypeName())));
                            builder.append(" ");
                            builder.append(((AssignmentExpression) exp).getOutputElement().getIdentifier());
                            builder.append("[];\n");
                        } else {
                            newGlobalAssignments.add(exp);
                        }
                    } else {
                        newGlobalAssignments.add(exp);
                    }
                }
            }
            AMT.setGlobalAssignments(newGlobalAssignments);
        }



        return builder.toString();
    }

    /**
     * Prints the declaration of the global variables.
     * @return
     */
    public String printGlobalDefines() {
        CppExpressionPrinter.setAMT(AMT);
        StringBuilder builder = new StringBuilder();
        for (IRLExpression exp: AMT.getGlobalAssignments() ) {
            if (exp instanceof AssignmentExpression) {
                if (((AssignmentExpression) exp).getOutputElement() instanceof PrimitiveData) {
                    OperationExpression value = ((AssignmentExpression) exp).getRhsExpression();
                    Data data = ((AssignmentExpression) exp).getOutputElement();
                    builder.append("#define ");
                    builder.append(data.getIdentifier());
                    builder.append(" (");
                    builder.append(CppExpressionPrinter.doPrintExpression(value, false, new ArrayList<>(), new ArrayList<>()));
                    builder.append(")\n");
                }
            }
        }
        return builder.toString();
    }

    /**
     * Prints the simple functions.
     * @return
     */
    public String printSimpleFunctions() {
        StringBuilder res = new StringBuilder();

        for (StringBuilder builder: CppPlainNodePrinter.getSimpleFunctions().values() ) {
            res.append(builder);
        }

        return res.toString();
    }

    /**
     * Prints the simple functions.
     * @return
     */
    public String printSimpleFunctionsCuda() {
        StringBuilder res = new StringBuilder();

        for (StringBuilder builder: CppPlainNodePrinter.getSimpleFunctionsCuda().values() ) {
            res.append(builder);
        }

        return res.toString();
    }
}