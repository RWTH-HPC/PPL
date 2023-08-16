package de.parallelpatterndsl.patterndsl.printer;

import de.parallelpatterndsl.patterndsl.CombinerFunction;
import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.CallGroup;
import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.ParallelGroup;
import de.parallelpatterndsl.patterndsl.MappingTree.GeneralDataPlacementFunctions.OffloadDataEncoding;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.ExtendedAMTShapeVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.*;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Node;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;
import de.se_rwth.commons.logging.Log;
import org.javatuples.Pair;
import org.javatuples.Quartet;

import java.util.*;


/**
 * This class implements a visitor, always starting from the root, which generates the source code from the given root.
 */
public class CppPlainNodePrinter implements ExtendedAMTShapeVisitor {

    /**
     * The set of currently active function calls, where the last call node describes the current scope.
     * This list is used to generate the correct variables.
     */
    private ArrayList<CallMapping> aktiveFunctions;

    /**
     * The set of currently active pattern calls, where the last pattern call node describes the current scope.
     * This list is used to generate the correct variables.
     */
    private ArrayList<ParallelCallMapping> activePatterns;

    /**
     * The current instance of this class.
     */
    private static CppPlainNodePrinter instance;

    /**
     * True, iff a parallel region is currently generated.
     */
    private boolean isCurrentlyParallel;

    /**
     * The hardware description of the target architecture.
     */
    private Network network;

    /**
     * The number of indents which is currently used.
     */
    private int numIndents;

    /**
     * True, iff the part will be executed on the GPU.
     */
    private boolean onGPU;

    /**
     * The APT to be generated.
     */
    private AbstractMappingTree AMT;

    /**
     * The builder used for generating the output.
     */
    private StringBuilder builder;

    /**
     * The builder used for generating the gpu source.
     */
    private StringBuilder gpuBuilder;

    /**
     * The builder used for generating the gpu header source.
     */
    private StringBuilder gpuHeaderBuilder;

    /**
     * The builder used for generating the gpu header source.
     */
    private StringBuilder gpuKernelHeaderBuilder;

    /**
     * Stores the simple functions which have been generated.
     */
    private HashMap<String, StringBuilder> simpleFunctions;

    /**
     * Stores the simple functions which have been generated for the GPU.
     */
    private HashMap<String, StringBuilder> simpleFunctionsGPU;

    /**
     * True, if the call needs to be inlined multiple times, used to avoid multiple instantiations of the same variable.
     */
    private boolean repetitiveLoopCall;

    /**
     * True, iff MPI is necessary.
     */
    private boolean needsMPI;

    /**
     * True, iff a current ancestor is a fused parallel call
     */
    private boolean currentlyFused;

    /**
     * A random string extension used by the current fused parallel call mapping.
     */
    private String fusedLambdaExtension;

    /**
     * Stores which scopes need to be closed after inlining.
     */
    private Stack<Pair<ReturnMapping, CallMapping>> scopeClosingStack;

    public CppPlainNodePrinter(AbstractMappingTree AMT, Network network) {
        aktiveFunctions = new ArrayList<>();
        activePatterns = new ArrayList<>();
        onGPU = false;
        builder = new StringBuilder();
        numIndents = 1;
        this.AMT = AMT;
        simpleFunctions = new HashMap<>();
        simpleFunctionsGPU = new HashMap<>();
        this.network = network;
        isCurrentlyParallel = false;
        needsMPI = network.getNodes().size() > 1;
        currentlyFused = false;
        fusedLambdaExtension = "";
        gpuHeaderBuilder = new StringBuilder();
        gpuBuilder = new StringBuilder();
        gpuKernelHeaderBuilder = new StringBuilder();
        CppExpressionPrinter.setNeedsMPI(needsMPI);
        scopeClosingStack = new Stack<>();
    }

    /**
     * Static implementation of the printer.
     * @param node
     * @param AMT
     * @return
     */
    public static Quartet<String,String,String,String> doPrintNode(FunctionMapping node, AbstractMappingTree AMT, Network network) {
        instance = new CppPlainNodePrinter(AMT, network);
        CppExpressionPrinter.setAMT(AMT);
        node.accept(instance);

        return new Quartet(instance.toString(), instance.toStringGPU(), instance.toStringGPUHeader(), instance.toStringGPUKernelHeader());
    }

    /*****************************************************************
     *
     *              Generator Functions
     *
     ******************************************************************/

    /**
     * Handles the variable instantiation in the main function.
     * @param node
     */
    @Override
    public void visit(MainMapping node) {
        for (IRLExpression exp : AMT.getGlobalAssignments()) {
            if (exp instanceof AssignmentExpression) {
            if (((AssignmentExpression) exp).getOutputElement() instanceof PrimitiveData) {
               continue;
            }
                if (((AssignmentExpression) exp).getOutputElement() instanceof ArrayData) {
                    if (((ArrayData) ((AssignmentExpression) exp).getOutputElement()).isOnStack()) {
                        continue;
                    }
                }
            }
            indent();
            builder.append(CppExpressionPrinter.doPrintExpression(exp,false, aktiveFunctions, activePatterns));
            builder.append(";\n");
        }
        addVariables(node);

        indent();
        builder.append("int NUM_CORES;\n");
        indent();
        builder.append("int NUM_GPUS;\n\n");

        for (int i = 0; i < network.getNodes().size(); i++) {
            Node currentNode = network.getNodes().get(i);
            if (needsMPI) {
                if (i == 0) {
                    indent();
                    builder.append("int rank, nprocs;\n");
                    indent();
                    builder.append("MPI_Status Stat;\n");
                    indent();
                    builder.append("MPI_Init(&argc, &argv);\n");
                    indent();
                    builder.append("MPI_Comm_size(MPI_COMM_WORLD, &nprocs);\n");
                    indent();
                    builder.append("MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n");
                }
                indent();
                builder.append("if (rank == ");
                builder.append(i);
                builder.append(" ) {\n");

                addIndent();
            }
            int cores = 0;
            for (Device device: currentNode.getDevices() ) {
                if (device.getType().equalsIgnoreCase("CPU")) {
                    for (Processor processor: device.getProcessor() ) {
                        cores += processor.getCores();
                    }
                }
            }
            indent();
            builder.append("NUM_CORES = ");
            builder.append(cores - 1);
            builder.append(";\n");

            int num_gpus = 0;
            for (Device device: currentNode.getDevices() ) {
                if (device.getType().equalsIgnoreCase("gpu")) {
                    num_gpus++;
                }
            }
            indent();
            builder.append("NUM_GPUS = ");
            builder.append(num_gpus);
            builder.append(";\n");

            if (needsMPI) {
                removeIndent();
                indent();
                builder.append("}");
                if (i == network.getNodes().size() - 1) {
                    builder.append("\n");
                } else {
                    builder.append(" else ");
                }
            }
        }
        indent();

        builder.append("std::vector<pthread_t> pthreads(0);\n");
        indent();
        builder.append("std::vector<Thread*> pool(NUM_CORES);\n");
        indent();
        builder.append("setPool(&pool,&pthreads);\n");
        indent();
        builder.append("std::vector<ThreadGPU> gpu_pool(NUM_GPUS);\n");
        indent();
        builder.append("setGPUPool(&gpu_pool);\n");
        indent();
        builder.append("startExecution();\n");
        indent();
        builder.append("startGPUExecution();\n");
    }

    @Override
    public void traverse(MainMapping node) {
        long duration = System.currentTimeMillis();
        int size = node.getChildren().size();
        for (int i = 0; i < size; i++) {
            MappingNode child = node.getChildren().get(i);
            if (i%200 == 0) {
                System.out.print("Gen Completion " + i + " of " + size + ": ");
                System.out.println(((double) i)/size * 100);
                System.out.println("Took " + ((double) (System.currentTimeMillis() - duration)/1000) + "s");
            }
            child.accept(getRealThis());
        }
    }

    private void startNodeAssignment(SerialNodeMapping node) {
        if (node.getParent() == AMT.getRoot() && needsMPI) {
            if (onGPU) {
                gpuIndent();
            } else {
                indent();
            }
            builder.append("if (rank == ");
            builder.append(node.getTargetNode().getRank());
            builder.append(") {\n");
            addIndent();
        }
    }

    private void endNodeAssignment(SerialNodeMapping node) {
        if (node.getParent() == AMT.getRoot() && needsMPI) {
            removeIndent();
            if (onGPU) {
                gpuIndent();
            } else {
                indent();
            }
            builder.append("}\n");
        }
    }

    /**
     * Creates the string for the simple expression block nodes when encountered.
     * @param node
     */
    @Override
    public void visit(SimpleExpressionBlockMapping node) {
        startNodeAssignment(node);
        boolean finalDefinition = false;
        for (IRLExpression exp: node.getExpressionList() ) {
            boolean doStackArrayInit = false;
            if (exp instanceof AssignmentExpression) {
                if (((AssignmentExpression) exp).getRhsExpression().getOperators().size() > 1) {
                    if (((AssignmentExpression) exp).getRhsExpression().getOperators().get(0) == Operator.LEFT_ARRAY_DEFINITION) {
                        endNodeAssignment(node);
                        doStackArrayInit = true;
                    }
                }
            }
            indent();
            builder.append(CppExpressionPrinter.doPrintExpression(exp,onGPU, aktiveFunctions, activePatterns));
            builder.append(";\n");
            if (doStackArrayInit && node.getExpressionList().get(node.getExpressionList().size() - 1) != exp) {
                startNodeAssignment(node);
            } else if (doStackArrayInit && node.getExpressionList().get(node.getExpressionList().size() - 1) == exp) {
                finalDefinition = true;
            }
        }
        if (!finalDefinition) {
            endNodeAssignment(node);
        }
    }

    /**
     * Creates the string for the complex expression node when encountered, also handles the inlining of functions.
     * @param node
     */
    @Override
    public void traverse(ComplexExpressionMapping node) {
        if (node.getExpression().hasProfilingInfo() && node.getParent() == AMT.getRoot()) {
            HashSet<Processor> allProcessors = new HashSet<>();
            for (Node computeNode : network.getNodes()) {
                for (Device device : computeNode.getDevices()) {
                    allProcessors.addAll(device.getProcessor());
                }
            }
            BarrierMapping barrierMapping = new BarrierMapping(Optional.of(AMT.getRoot()), node.getVariableTable(), allProcessors);
            barrierMapping.accept(this.getRealThis());
            if (needsMPI) {
                indent();
                builder.append("MPI_Barrier(MPI_COMM_WORLD);\n");
            }
        }
        if (!node.isArrayInitializer()) {
            startNodeAssignment(node);
        }

        for (MappingNode child: node.getChildren() ) {
            child.accept(this.getRealThis());
        }

        String expression = CppExpressionPrinter.doPrintExpression(node.getExpression(), onGPU, aktiveFunctions,activePatterns);
        if (!expression.isEmpty()) {
            indent();
            builder.append(CppExpressionPrinter.doPrintExpression(node.getExpression(), onGPU, aktiveFunctions, activePatterns));
            builder.append(";\n");
            ArrayList<String> closed = new ArrayList<>();
            while (!scopeClosingStack.isEmpty()) {
                if (node.getChildren().contains(scopeClosingStack.peek().getValue1())) {
                    Pair<ReturnMapping, CallMapping> scope = scopeClosingStack.pop();
                    closeInlineScope(scope.getValue0(), scope.getValue1());
                    if (scope.getValue0().getResult().isPresent()) {
                        Data returnData = ((OperationExpression) scope.getValue0().getResult().get().getExpression()).getOperands().get(0);
                        if (((SerialMapping) AbstractMappingTree.getFunctionTable().get(scope.getValue1().getFunctionIdentifier())).getShape().size() > 0) {
                            if (!((ArrayData) returnData).isOnStack()) {
                                String returnValue = scope.getValue1().getCallExpression().getIdentifier() + "_" + scope.getValue1().getCallExpression().getInlineEnding();
                                if (isClosed(closed, returnValue)) {
                                    indent();
                                    builder.append("std::free(");
                                    builder.append(returnValue);
                                    builder.append(");\n");
                                    closed.add(returnValue);
                                }
                            }
                        }
                    }
                } else {
                    break;
                }
            }


        }

        if (!node.isArrayInitializer()) {
            endNodeAssignment(node);
        }
    }

    private boolean isClosed(ArrayList<String> array, String test) {
        boolean res = true;

        for (String elem: array) {
            res &= !elem.equals(test);
        }

        return res;
    }

    /**
     * Creates the branches when encountered. Inlines functions in the branch conditions if necessary.
     * @param node
     */
    @Override
    public void traverse(BranchMapping node) {
        startNodeAssignment(node);
        for (int i = 0; i < node.getChildren().size(); i++) {
            BranchCaseMapping child = (BranchCaseMapping) node.getChildren().get(i);

            if (child.hasCondition()) {
                ComplexExpressionMapping condition = child.getCondition();
                for (MappingNode call: condition.getChildren() ) {
                    call.accept(this.getRealThis());
                }
            }
        }

        for (int i = 0; i < node.getChildren().size(); i++) {
            MappingNode child = node.getChildren().get(i);
            child.accept(this.getRealThis());
            if (i != node.getChildren().size() - 1) {
                builder.append(" else ");
            }
        }
        builder.append("\n");
        endNodeAssignment(node);
    }

    /**
     * Creates the header of a single branch.
     * @param node
     */
    @Override
    public void traverse(BranchCaseMapping node) {
        boolean open = true;

        if (node.hasCondition()) {
            indent();
            builder.append("if (");
            ComplexExpressionMapping condition =  node.getCondition();
            builder.append(CppExpressionPrinter.doPrintExpression(condition.getExpression(), onGPU, aktiveFunctions, activePatterns));
            builder.append(") {\n");

        } else {
            builder.append("{\n");
        }
        addIndent();
        addVariables(node);
        for (MappingNode child: node.getChildren()) {
            child.accept(this.getRealThis());
        }

        closeScope(node);

        removeIndent();
        indent();
        builder.append("}");

    }

    @Override
    public void visit(LoopSkipMapping node) {
        indent();
        if (node.isBreak()) {
            builder.append("break;\n");
        } else {
            builder.append("continue;\n");
        }
    }

    /**
     * Creates the header of a for each loop node. Handles the inlining of the assignment if necessary.
     * @param node
     */
    @Override
    public void traverse(ForEachLoopMapping node) {
        startNodeAssignment(node);
        indent();
        ComplexExpressionMapping init = node.getParsedList();

        for (MappingNode child: init.getChildren()) {
            child.accept(this.getRealThis());
        }

        node.setGenerationRandomIndex(RandomStringGenerator.getAlphaNumericString());

        builder.append(CppTypesPrinter.doPrintType(node.getLoopControlVariable().getTypeName()));
        builder.append("* ");
        builder.append("forEachLoopNodeArray_");
        builder.append(node.getGenerationRandomIndex());
        builder.append(" = ");
        builder.append(CppExpressionPrinter.doPrintExpression(init.getExpression(), onGPU, aktiveFunctions, activePatterns));
        builder.append(";\n");
        String loopCounter = "forEachLoopCounter_" + node.getGenerationRandomIndex();
        indent();
        builder.append("for ( size_t ");
        builder.append(loopCounter);
        builder.append(" = 0; ");
        builder.append(loopCounter);
        builder.append(" < ");
        builder.append(node.getNumIterations());
        builder.append("; ");
        builder.append(loopCounter);
        builder.append("++ ) {\n");

        addIndent();
        indent();
        builder.append(CppTypesPrinter.doPrintType(node.getLoopControlVariable().getTypeName()));
        builder.append("*");
        builder.append(" ");
        if (!(node.getLoopControlVariable() instanceof ArrayData)) {
            node.getLoopControlVariable().setIdentifier(node.getLoopControlVariable().getIdentifier() + "_"+ node.getGenerationRandomIndex());
        }
        builder.append(node.getLoopControlVariable().getIdentifier());
        builder.append(" = ");
        if (!(node.getLoopControlVariable() instanceof ArrayData)) {
            node.getLoopControlVariable().setIdentifier(node.getLoopControlVariable().getIdentifier() + "[0]");
        }
        builder.append("&forEachLoopNodeArray_");
        builder.append(node.getGenerationRandomIndex());
        builder.append("[");
        builder.append(loopCounter);
        if (node.getLoopControlVariable() instanceof ArrayData) {
            for (int size : ((ArrayData) node.getLoopControlVariable()).getShape()) {
                builder.append(" * ");
                builder.append(size);
            }
        }

        builder.append(" ];\n");

        addVariables(node);

        for (MappingNode child : node.getChildren()) {
            child.accept(this.getRealThis());
        }

        closeScope(node);

        node.getLoopControlVariable().setIdentifier(node.getLoopControlVariable().getIdentifier().split("_")[0]);

        removeIndent();
        indent();
        builder.append("}\n");
        endNodeAssignment(node);
    }

    /**
     * Creates the header of the for loop node. Handles the inlining of the initialization, update and condition in the beginning and for every iteration if necessary.
     * @param node
     */
    @Override
    public void traverse(ForLoopMapping node) {
        startNodeAssignment(node);
        ComplexExpressionMapping init = node.getInitExpression();
        ComplexExpressionMapping condition = node.getControlExpression();
        ComplexExpressionMapping update = node.getUpdateExpression();

        for (MappingNode child: init.getChildren()) {
            child.accept(this.getRealThis());
        }

        for (MappingNode child: condition.getChildren()) {
            child.accept(this.getRealThis());
        }

        for (MappingNode child: update.getChildren()) {
            child.accept(this.getRealThis());
        }
        indent();
        builder.append("for ( ");
        builder.append(CppTypesPrinter.doPrintType(node.getLoopControlVariable().getTypeName()));
        builder.append(" ");
        builder.append(CppExpressionPrinter.doPrintExpression(init.getExpression(), onGPU, aktiveFunctions, activePatterns));

        builder.append("; ");
        builder.append(CppExpressionPrinter.doPrintExpression(condition.getExpression(), onGPU, aktiveFunctions, activePatterns));

        builder.append("; ");
        builder.append(CppExpressionPrinter.doPrintExpression(update.getExpression(), onGPU, aktiveFunctions, activePatterns));
        builder.append(" ) {\n");

        addIndent();

        addVariables(node);

        for (MappingNode child: node.getChildren()) {
            child.accept(this.getRealThis());
        }

        // recompute the loop condition if necessary
        repetitiveLoopCall = true;
        for (MappingNode child: condition.getChildren()) {
            child.accept(this.getRealThis());
        }
        for (MappingNode child: update.getChildren()) {
            child.accept(this.getRealThis());
        }
        repetitiveLoopCall = false;

        closeScope(node);

        removeIndent();
        indent();
        builder.append("}\n");

        endNodeAssignment(node);
    }

    /**
     * Replaces the name of the indes variables with the current set of variables.
     * @param function
     * @param inlineEnding
     */
    private void replaceIndexNames(FunctionMapping function, String inlineEnding) {
        if (function instanceof RecursionMapping) {
            return;
        }
        function.getVariableTable().keySet().stream().filter( x -> x.startsWith("INDEX")).forEach(x -> function.getVariableTable().get(x).setIdentifier(x + "_" + inlineEnding));
    }

    /**
     * Creates the setup for the parallel execution in PThreads and CUDA.
     * @param node
     */
    @Override
    public void traverse(ParallelCallMapping node) {
        FunctionMapping function = AbstractMappingTree.getFunctionTable().get(node.getFunctionIdentifier());
        String inlineEnding = RandomStringGenerator.getAlphaNumericString();
        node.getCallExpression().setInlineEnding(inlineEnding);

        replaceIndexNames(function,inlineEnding);

        // generates a sequential version of the pattern.
        if (isCurrentlyParallel || node instanceof SerializedParallelCallMapping) {
            if (function instanceof MapMapping) {
                generateMapSerial(node, (MapMapping) function);
            } else if (function instanceof ReduceMapping) {
                generateReduceSerial(node, (ReduceMapping) function);
            } else if (function instanceof StencilMapping) {
                generateStencilSerial(node, (StencilMapping) function);
            } else if (function instanceof DynamicProgrammingMapping) {
                generateDPSerial(node, (DynamicProgrammingMapping) function);
            } else if (function instanceof RecursionMapping) {
                generateRecursionSerial(node, (RecursionMapping) function);
            } else {
                Log.error("Pattern of function: " + node.getFunctionIdentifier() + " not recognized!");
                throw new RuntimeException("Critical Error!");
            }
        } else {
            isCurrentlyParallel = true;
            if (function instanceof MapMapping) {
                //generateMapParallel(node, (MapMapping) function,!currentlyFused);
                generateMapParallel(node, (MapMapping) function,true);
            } else if (node instanceof ReductionCallMapping) {
                //generateReduceParallel((ReductionCallMapping) node, (ReduceMapping) function,!currentlyFused);
                generateReduceParallel((ReductionCallMapping) node, (ReduceMapping) function,true);
            } else if (function instanceof StencilMapping) {
                //generateStencilParallel(node, (StencilMapping) function,!currentlyFused);
                generateStencilParallel(node, (StencilMapping) function,true);
            } else if (function instanceof DynamicProgrammingMapping) {
                //generateDPParallel(node, (DynamicProgrammingMapping) function, !currentlyFused);
                generateDPParallel(node, (DynamicProgrammingMapping) function, true);
            } else if (function instanceof RecursionMapping) {
                generateRecursionParallel(node, (RecursionMapping) function);
            } else {
                Log.error("Pattern of function: " + node.getFunctionIdentifier() + " not recognized!");
                throw new RuntimeException("Critical Error!");
            }
            isCurrentlyParallel = false;
        }
    }

    /**
     * Adds the current call node to the list of active patterns.
     * @param node
     */
    @Override
    public void visit(ParallelCallMapping node) {
        activePatterns.add(node);
    }

    /**
     * Removes the current function from the list of active patterns.
     * @param node
     */
    @Override
    public void endVisit(ParallelCallMapping node) {
        activePatterns.remove(activePatterns.size() - 1);
    }

    /**
     * Adds the current call node to the list of active functions.
     * @param node
     */
    @Override
    public void visit(CallMapping node) {
        aktiveFunctions.add(node);
    }

    @Override
    public void visit(FusedParallelCallMapping node) {
        currentlyFused = true;
        fusedLambdaExtension = RandomStringGenerator.getAlphaNumericString();
        String lambda = "f_" + fusedLambdaExtension;
        //createFusedLambdaHeader(lambda,node);
    }

    @Override
    public void endVisit(FusedParallelCallMapping node) {
        currentlyFused = false;
        //createFusedLambdaEnd(node);
    }

    public static boolean calculateInlining(CallMapping call) {
        SerialMapping function = (SerialMapping) AbstractMappingTree.getFunctionTable().get(call.getFunctionIdentifier());
        boolean doInline = false;
        for (Data parameter: function.getArgumentValues() ) {
            if (parameter instanceof ArrayData || function.isHasParallelDescendants()) {
                doInline = true;
                break;
            }
        }

        if (function.isList()) {
            doInline = true;
        }
        return doInline;
    }

    /**
     * Generates the call node and inlines the function if necessary.
     * @param node
     */
    @Override
    public void traverse(CallMapping node) {
        SerialMapping function = (SerialMapping) AbstractMappingTree.getFunctionTable().get(node.getFunctionIdentifier());
        boolean doInline = calculateInlining(node);

        boolean hasExpression = true;
        if (function instanceof SerialMapping) {
            if (((SerialMapping) function).getReturnType() == PrimitiveDataTypes.VOID) {
                hasExpression = false;
            }
        }

        if (doInline || onGPU) {
            node.getCallExpression().setInlineEnding(RandomStringGenerator.getAlphaNumericString());

            // generate the return value
            if (!repetitiveLoopCall && hasExpression) {
                indent();
                builder.append(CppTypesPrinter.doPrintType(node.getCallExpression().getTypeName()));
                if (!node.getCallExpression().getShape().isEmpty()) {
                    builder.append("*");
                }
                builder.append(" ");

                builder.append(node.getCallExpression().getIdentifier());
                builder.append("_");
                builder.append(node.getCallExpression().getInlineEnding());
                builder.append(";\n");

                addVariables(node);
            }

            indent();
            builder.append("{\n");
            addIndent();

            // set arguments to parameter (create inlined variable)
            for (int i = 0; i < function.getArgumentValues().size(); i++) {
                Data parameter = function.getArgumentValues().get(i);
                indent();
                //only generate parameters, on the first generation.
                if (!repetitiveLoopCall) {
                    builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                    if (parameter instanceof ArrayData) {
                        builder.append("*");
                    }
                    builder.append(" ");
                }
                builder.append(parameter.getIdentifier());
                builder.append("_");
                builder.append(node.getCallExpression().getInlineEnding());

                builder.append(" = ");
                if (parameter instanceof ArrayData) {
                    builder.append("copy(");

                    // Test if a reference is necessary
                    boolean doRef = false;
                    boolean hasAccess = false;
                    if (!node.getArgumentExpressions().get(i).getOperands().get(0).getIdentifier().endsWith("]")) {
                        doRef = true;
                    }
                    if (node.getArgumentExpressions().get(i).getOperators().size() > 0) {
                        if (node.getArgumentExpressions().get(i).getOperators().get(0) == Operator.LEFT_ARRAY_ACCESS ) {
                            hasAccess = true;
                        }
                    }
                    if (doRef && hasAccess) {
                        builder.append("&");
                    }
                }
                builder.append(CppExpressionPrinter.doPrintExpression(node.getArgumentExpressions().get(i), onGPU, new ArrayList<>(aktiveFunctions.subList(0,aktiveFunctions.size()-1)), activePatterns));
                if (parameter instanceof ArrayData) {
                    builder.append(", 1LL");
                    for (int dimension: node.getArgumentExpressions().get(i).getShape()) {
                        builder.append(" * ");
                        builder.append(dimension);
                        builder.append("LL");
                    }
                    builder.append(")");
                }
                builder.append(";\n");
            }

            for (MappingNode child: node.getChildren()) {
                child.accept(getRealThis());
            }

            indent();
            builder.append("STOP_LABEL_");
            builder.append(node.getCallExpression().getInlineEnding());
            builder.append(":\n");
            indent();
            builder.append("noop;\n");

            removeIndent();
            indent();
            builder.append("}\n");
        } else {
            addSimpleFunction(node, function);
            addSimpleFunctionGPU(node, function);
        }
    }

    /**
     * Removes the current function from the list of active functions.
     * @param node
     */
    @Override
    public void endVisit(CallMapping node) {
        aktiveFunctions.remove(aktiveFunctions.size() - 1);
    }

    /**
     * Generates the the return node and handles inlining if necessary.
     * @param node
     */
    @Override
    public void traverse(ReturnMapping node) {
        MappingNode parent = node.getParent();
        boolean hasExpression = true;
        while (!(parent instanceof FunctionMapping)) {
            parent = parent.getParent();
        }

        if (parent instanceof SerialMapping) {
            if (((SerialMapping) parent).getReturnType() == PrimitiveDataTypes.VOID) {
                hasExpression = false;
            }
        }
        ComplexExpressionMapping expNode = null;
        if (hasExpression) {
            expNode = node.getResult().get();

            for (MappingNode child : expNode.getChildren()) {
                child.accept(this.getRealThis());
            }
        }

        if (node.getParent() instanceof MainMapping) {
            indent();
            builder.append("finishExecution();\n");
            indent();
            builder.append("finishGPUExecution();\n");
            closeScope(node);
            if (needsMPI) {
                indent();
                builder.append("MPI_Finalize();\n");
            }
        }

        indent();
        if (aktiveFunctions.isEmpty()) {
            builder.append("return ");
            if (hasExpression) {
                builder.append(CppExpressionPrinter.doPrintExpression(expNode.getExpression(), onGPU, aktiveFunctions, activePatterns));
            }
            builder.append(";\n");
        } else {
            CallMapping call = aktiveFunctions.get(aktiveFunctions.size() - 1);

            SerialMapping function = (SerialMapping) AbstractMappingTree.getFunctionTable().get(call.getFunctionIdentifier());
            boolean doInline = calculateInlining(call);

            boolean doRef = false;
            boolean hasAccess = false;
            if (doInline || onGPU ) {
                if (hasExpression) {
                    builder.append(call.getCallExpression().getIdentifier());
                    builder.append("_");
                    builder.append(call.getCallExpression().getInlineEnding());
                    builder.append(" = ");

                    // generate reference assignment
                    if (!((OperationExpression) expNode.getExpression()).getOperands().get(0).getIdentifier().endsWith("]")) {
                        doRef = true;
                    }
                    if (((OperationExpression) expNode.getExpression()).getOperators().size() > 0) {
                        if (((OperationExpression) expNode.getExpression()).getOperators().get(0) == Operator.LEFT_ARRAY_ACCESS ) {
                            hasAccess = true;
                        }
                    }
                    if (doRef && hasAccess && !expNode.getExpression().getShape().isEmpty()) {
                        builder.append("copy(&");
                    }
                }
            } else {
                closeScope(node);
                indent();
                builder.append("return ");
            }
            if (hasExpression) {
                builder.append(CppExpressionPrinter.doPrintExpression(expNode.getExpression(), onGPU, aktiveFunctions, activePatterns));
                if (doInline || onGPU ) {
                    if (doRef && hasAccess && !expNode.getExpression().getShape().isEmpty()) {
                        builder.append(", ");
                        for (Integer dim: expNode.getExpression().getShape() ) {
                            builder.append(dim);
                            builder.append(" * ");
                        }
                        builder.append(" 1, true)");
                    }
                }
            }
            builder.append(";\n");
            if (doInline || onGPU) {
                scopeClosingStack.push(new Pair<>(node, aktiveFunctions.get(aktiveFunctions.size() - 1)));
                indent();
                builder.append("goto STOP_LABEL_");
                builder.append(call.getCallExpression().getInlineEnding());
                builder.append(";\n");
            }
        }
    }

    /**
     * Generates the header of the current while loop node and inlines functions in the condition if necessary.
     * @param node
     */
    @Override
    public void traverse(WhileLoopMapping node) {
        startNodeAssignment(node);
        ComplexExpressionMapping condition = node.getCondition();

        for (MappingNode child: condition.getChildren()) {
            child.accept(this.getRealThis());
        }
        indent();
        builder.append("while ( ");
        builder.append(CppExpressionPrinter.doPrintExpression(condition.getExpression(), onGPU, aktiveFunctions, activePatterns));
        builder.append(" ) {\n");

        addIndent();

        addVariables(node);

        for (MappingNode child:  node.getChildren()) {
            child.accept(this.getRealThis());
        }

        // recompute the loop condition if necessary
        repetitiveLoopCall = true;
        for (MappingNode child: condition.getChildren()) {
            child.accept(this.getRealThis());
        }
        repetitiveLoopCall = false;

        closeScope(node);

        removeIndent();
        indent();
        builder.append("}\n");
        endNodeAssignment(node);
    }

    @Override
    public void visit(JumpLabelMapping node) {
        indent();
        builder.append("STOP_LABEL_");
        builder.append(node.getLabel());
        builder.append(":\n");
    }

    @Override
    public void visit(JumpStatementMapping node) {
        for (Data data : node.getClosingVars()) {
            if ((!AMT.getGlobalVariableTable().values().contains(data) && !data.isParameter() && !data.isReturnData()) && !(data instanceof FunctionInlineData)) {
                if (((ArrayData) data).isOnStack()) {
                    continue;
                }
                ArrayList<Data> globalScope = new ArrayList<>(AMT.getGlobalVariableTable().values());
                if (!((ArrayData) data).isOnStack() && !globalScope.contains(data) && !data.isParameter() && !data.getIdentifier().startsWith("inlineReturn_") && !data.isInlinedParameter() && !data.isInlinedReturnValue() && !((OperationExpression) node.getResultExpression().getExpression()).getOperands().contains(data)) {
                    if(!data.isClosed()) {
                        indent();
                        builder.append("std::free(");
                        builder.append(data.getIdentifier());
                        builder.append(");\n");
                        data.setClosed();
                    }
                }
            }
        }
        indent();
        builder.append(node.getReturnOutputData().getIdentifier());
        builder.append(" = ");
        builder.append(CppExpressionPrinter.doPrintExpression(node.getResultExpression().getExpression(), onGPU, aktiveFunctions, activePatterns));
        builder.append(";\n");

        indent();
        builder.append("goto ");
        builder.append(node.getLabel());
        builder.append(";\n");
    }

    @Override
    public void traverse(JumpStatementMapping node) {

    }

    @Override
    public void visit(BarrierMapping node) {
        HashSet<Processor> toSynchronize;
        HashMap<Device,HashSet<Processor>> CPUsync = new HashMap<>();
        HashMap<Node,HashSet<Device>> GPUsync = new HashMap<>();

        // instantiate the two different Barrier definitions.
        if (node.isGroupBased()) {
            toSynchronize = new HashSet<>();
            for (ParallelGroup group: node.getBarrier() ) {
                toSynchronize.addAll(group.getProcessors());
            }
        } else {
            toSynchronize = new HashSet<>(node.getBarrierProc());
        }

        for (Processor proc: toSynchronize ) {
            if (proc.getParent().getType().equalsIgnoreCase("gpu")) {
                if (!GPUsync.containsKey(proc.getParent().getParent())) {
                    GPUsync.put(proc.getParent().getParent(), new HashSet<>());
                }
                GPUsync.get(proc.getParent().getParent()).add(proc.getParent());
            } else {
                if (!CPUsync.containsKey(proc.getParent())) {
                    CPUsync.put(proc.getParent(), new HashSet<>());
                }
                CPUsync.get(proc.getParent()).add(proc);
            }
        }

        // generate CPU synchronization
        for (Device CPU: CPUsync.keySet()) {
            if (needsMPI) {
                indent();
                builder.append("if (rank == ");
                builder.append(CPU.getParent().getRank());
                builder.append(") {\n");
                addIndent();
            }
            String maskName = "mask_ptr_" + RandomStringGenerator.getAlphaNumericString();
            String sharedMaskName = "boost_" + maskName;
            indent();
            builder.append("Bit_Mask * ");
            builder.append(maskName);
            builder.append(" = new Bit_Mask(");
            builder.append(CPU.getProcessor().size() * CPU.getProcessor().get(0).getCores() - 1);
            if (CPU.getProcessor().size() == CPUsync.get(CPU).size()) {
                builder.append(",false);\n");
            } else {
                builder.append(",true);\n");

                for (Processor proc : CPUsync.get(CPU)) {
                    int offset = 1;
                    if (proc.isFirstCG()) {
                        offset = 0;
                    }

                    indent();
                    String iterName = "i_" + RandomStringGenerator.getAlphaNumericString();
                    builder.append("for (size_t ");
                    builder.append(iterName);
                    builder.append(" = ");
                    builder.append(proc.getCores() * proc.getRank() - offset);
                    builder.append("; ");
                    builder.append(iterName);
                    builder.append(" < ");
                    builder.append(proc.getCores() * (proc.getRank() + 1) - 1);
                    builder.append("; ++");
                    builder.append(iterName);
                    builder.append(") {\n");
                    addIndent();
                    indent();
                    builder.append(maskName);
                    builder.append("->setBarrier(");
                    builder.append(iterName);
                    builder.append(");\n");
                    removeIndent();
                    indent();
                    builder.append("}\n");
                }
            }

            indent();
            builder.append("boost::shared_ptr<Bit_Mask>");
            builder.append(sharedMaskName);
            builder.append(" (");
            builder.append(maskName);
            builder.append(");\n");
            indent();
            builder.append("self_barrier(");
            builder.append(sharedMaskName);
            builder.append(");\n");
            if (needsMPI) {
                removeIndent();
                indent();
                builder.append("}\n");
            }
        }

        // generate GPU synchronization
        for (Node computeNode: GPUsync.keySet() ) {
            if (needsMPI) {
                indent();
                builder.append("if (rank == ");
                builder.append(computeNode.getRank());
                builder.append(") {\n");
                addIndent();
            }
            int numGPUs = 0;
            for (Device device: computeNode.getDevices() ) {
                if (device.getType().equalsIgnoreCase("gpu")) {
                    numGPUs++;
                }
            }
            String maskName = "mask_ptr_" + RandomStringGenerator.getAlphaNumericString();
            String sharedMaskName = "boost_" + maskName;
            indent();
            builder.append("Bit_Mask * ");
            builder.append(maskName);
            builder.append(" = new Bit_Mask(");
            builder.append(numGPUs);

            if (numGPUs == GPUsync.get(computeNode).size()) {
                builder.append(",false);\n");
            } else {
                builder.append(",true);\n");

                for (Device device : GPUsync.get(computeNode)) {
                    indent();
                    builder.append(maskName);
                    builder.append("->setBarrier(");
                    builder.append(device.getGPUrank());
                    builder.append(");\n");
                }
            }

            indent();
            builder.append("boost::shared_ptr<Bit_Mask>");
            builder.append(sharedMaskName);
            builder.append(" (");
            builder.append(maskName);
            builder.append(");\n");
            indent();
            builder.append("cuda_sync_device(");
            builder.append(sharedMaskName);
            builder.append(");\n");
            if (needsMPI) {
                removeIndent();
                indent();
                builder.append("}\n");
            }
        }

    }

    @Override
    public void visit(DataMovementMapping node) {
        // automatically copy data from GPUs not necessary
        ArrayList<DataPlacement> reducedPlacementsSender = new ArrayList<>();
        for (DataPlacement placement: node.getSender() ) {
            for (EndPoint end: placement.getPlacement() ) {
                if (end.getLocation().getType().equalsIgnoreCase("gpu")) {
                    end.setLocation(end.getLocation().getParent().getDevices().get(0));
                }
            }
            reducedPlacementsSender.add(new DataPlacement(placement.getPlacement(), placement.getDataElement()));
        }
        ArrayList<DataPlacement> reducedPlacementsReceiver = new ArrayList<>();
        for (DataPlacement placement: node.getReceiver() ) {
            for (EndPoint end: placement.getPlacement() ) {
                if (end.getLocation().getType().equalsIgnoreCase("gpu")) {
                    end.setLocation(end.getLocation().getParent().getDevices().get(0));
                }
            }
            reducedPlacementsReceiver.add(new DataPlacement(placement.getPlacement(), placement.getDataElement()));
        }

        /*
        Generating the data transfers, we currently assume, that all CPUs on one node can access the main memory and do not need to send data.
         */
        for (DataPlacement placement: reducedPlacementsReceiver ) {
            Optional<DataPlacement> OPTSender = reducedPlacementsSender.stream().filter(x -> x.getDataElement() == placement.getDataElement()).findFirst();
            DataPlacement sender;
            if (OPTSender.isPresent()) {
                sender = OPTSender.get();
            } else {
                Log.error("No sending data found for " + placement.getDataElement().getIdentifier());
                throw new RuntimeException("Critical Error!");
            }

            for (EndPoint destination: placement.getPlacement() ) {
                ArrayList<EndPoint> overlap = getOverlap(sender.getPlacement(), destination);
                long currentStart = destination.getStart();
                long currentLength = destination.getLength();
                for (EndPoint target: overlap ) {
                    if (target.getStart() > currentStart) {
                        if (target.getLocation() == destination.getLocation()) {
                            continue;
                        }
                        Log.error("Partially non existent data for " + placement.getDataElement().getIdentifier());
                        throw new RuntimeException("Critical Error!");
                    }
                    if (target.getStart() + target.getLength() < currentStart) {
                        continue;
                    }
                    if (target.getLocation().getParent() == destination.getLocation().getParent()) {
                        if (target.getStart() + target.getLength() - currentStart >= currentLength) {
                            break;
                        } else {
                            currentLength = currentLength - (target.getStart() + target.getLength() - currentStart);
                            currentStart = target.getStart() + target.getLength();
                        }
                    } else {
                        long numElements = 1;
                        if (placement.getDataElement() instanceof ArrayData) {
                            for (int i = 1; i < ((ArrayData) placement.getDataElement()).getShape().size(); i++) {
                                numElements *= ((ArrayData) placement.getDataElement()).getShape().get(i);
                            }
                        }
                        if (target.getStart() + target.getLength() - currentStart >= currentLength) {
                            MPITransferGeneration(placement.getDataElement(), currentStart * numElements, currentLength * numElements, target, destination);
                            break;
                        } else {
                            long reducedLength = target.getStart() + target.getLength() - currentStart;
                            MPITransferGeneration(placement.getDataElement(), currentStart * numElements, reducedLength * numElements, target, destination);
                            currentLength = currentLength - reducedLength;
                            currentStart = target.getStart() + target.getLength();
                        }
                    }
                }
            }
        }
    }


    @Override
    public void visit(GPUAllocationMapping node) {
        indent();
        builder.append(CppTypesPrinter.doPrintType(node.getAllocator().getData().getTypeName()));
        builder.append("* ");
        builder.append(node.getAllocator().getIdentifier());
        builder.append(";\n");

        if (needsMPI) {
            indent();
            builder.append("if (rank == ");
            builder.append(node.getAllocator().getDevice().getParent().getRank());
            builder.append(") {\n");
            addIndent();
        }

        String lambda = "f_alloc_" + RandomStringGenerator.getAlphaNumericString();
        createAllocLambdaHeader(lambda, node.getAllocator());

        indent();
        CppGPUDataNodePrinter.generateGPUAllocMapping(node,builder);

        createGPULambdaEnd(lambda, node.getAllocator().getDevice().getGPUrank());

        if (needsMPI) {
            removeIndent();
            indent();
            builder.append("}\n");
        }
    }

    @Override
    public void visit(GPUDeAllocationMapping node) {
        if (needsMPI) {
            indent();
            builder.append("if (rank == ");
            builder.append(node.getAllocator().getDevice().getParent().getRank());
            builder.append(") {\n");
            addIndent();
        }

        String lambda = "f_dealloc_" + RandomStringGenerator.getAlphaNumericString();
        createAllocLambdaHeader(lambda, node.getAllocator());

        indent();
        CppGPUDataNodePrinter.generateGPUDeAllocMapping(node, builder);

        createGPULambdaEnd(lambda, node.getAllocator().getDevice().getGPUrank());

        if (needsMPI) {
            removeIndent();
            indent();
            builder.append("}\n");
        }
    }

    @Override
    public void visit(GPUDataMovementMapping node) {
        if (needsMPI) {
            indent();
            builder.append("if (rank == ");
            builder.append(node.getAllocator().getDevice().getParent().getRank());
            builder.append(") {\n");
            addIndent();
        }

        String lambda = "f_movement_" + RandomStringGenerator.getAlphaNumericString();
        createMovementLambdaHeader(lambda, node.getAllocator());

        indent();
        CppGPUDataNodePrinter.generateGPUDataMovementMapping(node, builder);

        createGPULambdaEnd(lambda, node.getAllocator().getDevice().getGPUrank());

        if (needsMPI) {
            removeIndent();
            indent();
            builder.append("}\n");
        }
    }

    /**
     * Generates the string necessary for MPI communication between source and destination.
     * @param data
     * @param start
     * @param size
     * @param target
     * @param destination
     */
    private void MPITransferGeneration(Data data, long start, long size, EndPoint target, EndPoint destination) {
        indent();
        builder.append("if (rank == ");
        builder.append(target.getLocation().getParent().getRank());
        builder.append(") {\n");
        addIndent();
        indent();
        //Send
        builder.append("MPI_Send(&");
        builder.append(data.getIdentifier());
        if (data instanceof ArrayData) {
            builder.append("[");
            builder.append(start);
            builder.append("]");
        }

        builder.append(", ");
        builder.append(size);
        builder.append(", ");
        builder.append(MPITypesPrinter.doPrintType(data.getTypeName()));
        builder.append(", ");
        builder.append(destination.getLocation().getParent().getRank());
        builder.append(", 0, MPI_COMM_WORLD);\n");
        removeIndent();
        indent();

        // Receive
        builder.append("} else if (rank == ");
        builder.append(destination.getLocation().getParent().getRank());
        builder.append(") {\n");
        addIndent();
        indent();
        builder.append("MPI_Recv(&");
        builder.append(data.getIdentifier());
        if (data instanceof ArrayData) {
            builder.append("[");
            builder.append(start);
            builder.append("]");
        }

        builder.append(", ");
        builder.append(size);
        builder.append(", ");
        builder.append(MPITypesPrinter.doPrintType(data.getTypeName()));
        builder.append(", ");
        builder.append(target.getLocation().getParent().getRank());
        builder.append(", 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);\n");
        removeIndent();
        indent();
        builder.append("}\n");
    }

    /**
     * Returns a sublist of EndPoints that overlap with element. May be optimized in the future.
     * @param list
     * @param element
     * @return
     */
    private ArrayList<EndPoint> getOverlap(ArrayList<EndPoint> list, EndPoint element) {
        ArrayList<EndPoint> result = new ArrayList<>();
        for (EndPoint testing: list ) {
            if (testing.getStart() <= element.getStart() && testing.getStart() + testing.getLength() > element.getStart()) {
                result.add(testing);
            } else if (testing.getStart() + testing.getLength() >= element.getStart() + element.getLength() && testing.getStart() < element.getStart() + element.getLength()) {
                result.add(testing);
            } else if (testing.getStart() > element.getStart() && testing.getStart() + testing.getLength() < element.getStart() + element.getLength()) {
                result.add(testing);
            }
        }
        return result;
    }

    /*****************************************************************
     *
     *              Parallel Generation Functions
     *
     ******************************************************************/

    /**
     * Creates an operation expression from the left side of an assignment expression.
     * @param assignmentExpression
     * @return
     */
    private OperationExpression lhs2OperationExpression(AssignmentExpression assignmentExpression) {
        ArrayList<Data> operands = new ArrayList<>();
        ArrayList<Operator> operators = new ArrayList<>();

        operands.add(assignmentExpression.getOutputElement());
        for (OperationExpression opexp: assignmentExpression.getAccessScheme()) {
            operators.add(Operator.LEFT_ARRAY_ACCESS);
            operands.addAll(opexp.getOperands());
            operators.addAll(opexp.getOperators());
            operators.add(Operator.RIGHT_ARRAY_ACCESS);
        }
        return new OperationExpression(operands,operators);
    }

    /**
     * Fills the ParallelVariableInliningTable for all parameter within the pattern call.
     * @param call
     * @param function
     */
    private void fillParallelVariableInliningTable(ParallelCallMapping call, ParallelMapping function) {
        ComplexExpressionMapping complexExpressionMapping = call.getDefinition();
        AssignmentExpression assignmentExpression = (AssignmentExpression) complexExpressionMapping.getExpression();


        CppExpressionPrinter.addParallelVariableInlining(function.getReturnElement(), lhs2OperationExpression(assignmentExpression));

        for (int i = 0; i < function.getArgumentValues().size(); i++) {
            CppExpressionPrinter.addParallelVariableInlining(function.getArgumentValues().get(i), call.getArgumentExpressions().get(i));
        }
    }

    /**
     * Clears the ParallelVariableInliningTable for all parameters within the closing parallel node.
     * @param function
     */
    private void clearParallelVariableInliningTable(ParallelMapping function) {
        CppExpressionPrinter.removeParallelVariableInlining(function.getReturnElement());

        for (Data data: function.getArgumentValues() ) {
            CppExpressionPrinter.removeParallelVariableInlining(data);
        }
    }

    /**
     * Generates a serial version of the map pattern.
     * @param call
     * @param function
     */
    private void generateMapSerial(ParallelCallMapping call, MapMapping function) {
        fillParallelVariableInliningTable(call,function);

        // length and start are defined as the first and second additional arguments by the language standard.
        long length = call.getNumIterations().get(0);
        long start = call.getStartIndex().get(0);
        String index = "INDEX_" + call.getCallExpression().getInlineEnding();

        indent();
        builder.append("for (size_t ");
        builder.append(index);
        builder.append(" = ");
        builder.append(start);
        builder.append("; ");
        builder.append(index);
        builder.append(" < ");
        builder.append(start);
        builder.append(" + ");
        builder.append(length);
        builder.append("; ++");
        builder.append(index);
        builder.append(") {\n");
        addIndent();

        addVariables(function);

        for (MappingNode child: function.getChildren()) {
            child.accept(getRealThis());
        }

        closeScope(function);

        removeIndent();
        indent();
        builder.append("}\n");
        clearParallelVariableInliningTable(function);
    }

    /**
     * Generates a serial version of the reduction pattern.
     * @param call
     * @param function
     */
    private void generateReduceSerial(ParallelCallMapping call, ReduceMapping function) {
        fillParallelVariableInliningTable(call,function);

        // length and start are defined as the first and second additional arguments by the language standard.
        long length = call.getNumIterations().get(0);
        long start = call.getStartIndex().get(0);
        String index = "INDEX_" + call.getCallExpression().getInlineEnding();

        indent();
        builder.append("for (size_t ");
        builder.append(index);
        builder.append(" = ");
        builder.append(start);
        builder.append("; ");
        builder.append(index);
        builder.append(" < ");
        builder.append(start);
        builder.append(" + ");
        builder.append(length);
        builder.append("; ++");
        builder.append(index);
        builder.append(") {\n");
        addIndent();

        addVariables(function);

        for (MappingNode child: function.getChildren()) {
            child.accept(getRealThis());
        }

        closeScope(function);

        removeIndent();
        indent();
        builder.append("}\n");
        clearParallelVariableInliningTable(function);
    }

    /**
     * Generates a serial version of the stencil pattern.
     * @param call
     * @param function
     */
    private void generateStencilSerial(ParallelCallMapping call, StencilMapping function) {
        fillParallelVariableInliningTable(call,function);

        ArrayList<Long> lengths = call.getNumIterations();
        ArrayList<Long> starts = call.getStartIndex();
        int dimensions = function.getDimension();

        for (int i = 0; i < dimensions; i++) {
            String index = "INDEX" + i + "_" + call.getCallExpression().getInlineEnding();

            indent();
            builder.append("for (size_t ");
            builder.append(index);
            builder.append(" = ");
            builder.append(starts.get(i));
            builder.append("; ");
            builder.append(index);
            builder.append(" < ");
            builder.append(starts.get(i));
            builder.append(" + ");
            builder.append(lengths.get(i));
            builder.append("; ++");
            builder.append(index);
            builder.append(") {\n");
            addIndent();
        }

        addVariables(function);

        for (MappingNode child: function.getChildren()) {
            child.accept(getRealThis());
        }

        closeScope(function);

        for (int i = 0; i < dimensions; i++) {
            removeIndent();
            indent();
            builder.append("}\n");
        }

        clearParallelVariableInliningTable(function);
    }

    /**
     * Generates a serial version of the dynamic programming pattern.
     * @param call
     * @param function
     */
    private void generateDPSerial(ParallelCallMapping call, DynamicProgrammingMapping function) {
        fillParallelVariableInliningTable(call,function);

        ArrayList<Long> lengths = call.getNumIterations();
        ArrayList<Long> starts = call.getStartIndex();
        int dimensions = lengths.size();

        for (int i = 0; i < dimensions; i++) {
            String index = "INDEX" + i + "_" + call.getCallExpression().getInlineEnding();

            indent();
            builder.append("for (size_t ");
            builder.append(index);
            builder.append(" = ");
            builder.append(starts.get(i));
            builder.append("; ");
            builder.append(index);
            builder.append(" < ");
            builder.append(starts.get(i));
            builder.append(" + ");
            builder.append(lengths.get(i));
            builder.append("; ++");
            builder.append(index);
            builder.append(") {\n");
            addIndent();
        }

        addVariables(function);

        for (MappingNode child: function.getChildren()) {
            child.accept(getRealThis());
        }


        removeIndent();
        indent();
        builder.append("}\n");

        indent();
        builder.append("Set_Partial_Array( ");
        if (CppExpressionPrinter.getParallelVariableInlining(function.getInputElements().get(0)).getOperators().size() > 0) {
            builder.append("&");
        }
        builder.append(CppExpressionPrinter.doPrintExpression(CppExpressionPrinter.getParallelVariableInlining(function.getInputElements().get(0)), onGPU, aktiveFunctions, activePatterns));
        builder.append(", ");
        if (CppExpressionPrinter.getParallelVariableInlining(function.getOutputElements().get(0)).getOperators().size() > 0) {
            builder.append("&");
        }
        builder.append(CppExpressionPrinter.doPrintExpression(CppExpressionPrinter.getParallelVariableInlining(function.getOutputElements().get(0)), onGPU, aktiveFunctions, activePatterns));
        builder.append(", ");
        if (call.getOutputElements().get(0) instanceof ArrayData) {
            builder.append(((ArrayData) call.getOutputElements().get(0)).getShape().get(0));
        } else {
            builder.append("0");
        }
        builder.append(");\n");
        closeScope(function);
        removeIndent();
        indent();
        builder.append("}\n");


        clearParallelVariableInliningTable(function);
    }

    /**
     * Generates a serial version of the recursion pattern.
     * @param call
     * @param function
     */
    private void generateRecursionSerial(ParallelCallMapping call, RecursionMapping function) {
        indent();
        builder.append(CppExpressionPrinter.doPrintExpression(call.getDefinition().getExpression(), onGPU, aktiveFunctions, activePatterns));
        builder.append(";\n");
        addSimpleFunction(call, function);
        addSimpleFunctionGPU(call, function);
    }

    /**
     * Generates the lambda header for parallel call nodes on the CPU.
     * @param lambda
     * @param data
     */
    private void createAllocLambdaHeader(String lambda, OffloadDataEncoding data) {
        indent();
        builder.append("auto ");
        builder.append(lambda);
        builder.append(" = [");
        builder.append("&");
        //builder.append(data.getIdentifier());
        builder.append("] () {\n");
        addIndent();
    }

    /**
     * Generates the lambda header for parallel call nodes on the CPU.
     * @param lambda
     * @param data
     */
    private void createMovementLambdaHeader(String lambda, OffloadDataEncoding data) {
        indent();
        builder.append("auto ");
        builder.append(lambda);
        builder.append(" = [");
        builder.append("&");
        /*builder.append(data.getIdentifier());
        builder.append(", &");
        builder.append(data.getData().getIdentifier());*/
        builder.append("] () {\n");
        addIndent();
    }

    /**
     * Generates the lambda header for parallel call nodes on the CPU.
     * @param lambda
     * @param call
     */
    private void createLambdaHeader(String lambda, ParallelCallMapping call) {
        indent();
        builder.append("auto ");
        builder.append(lambda);
        builder.append(" = [");

        builder.append("&");
        /*
        // create header for GPU calls
        if (call instanceof GPUParallelCallMapping) {
            for (OffloadDataEncoding encoding: ((GPUParallelCallMapping) call).getInputDataEncodings() ) {
                if (((GPUParallelCallMapping) call).getOutputDataEncodings().stream().map(x -> x.getData()).allMatch(x -> x != encoding.getData())) {
                    if (!(encoding.getData() instanceof LiteralData)) {
                        if (encoding.getData() instanceof PrimitiveData) {
                            builder.append("&");
                        }
                        builder.append(encoding.getIdentifier());
                        builder.append(", ");
                    }
                }
            }
            for (OffloadDataEncoding encoding: ((GPUParallelCallMapping) call).getOutputDataEncodings() ) {
                builder.append("&");

                builder.append(encoding.getIdentifier());
                builder.append(", ");
            }

            builder.deleteCharAt(builder.length() - 2);
            builder.append("] () {\n");
            addIndent();
            return;
        } else if (call instanceof ReductionCallMapping) {
            if (((ReductionCallMapping) call).getOnGPU()) {
                for (OffloadDataEncoding encoding : ((ReductionCallMapping) call).getInputDataEncodings()) {
                    if (!(encoding.getData() instanceof LiteralData)) {
                        if (encoding.getData() instanceof PrimitiveData) {
                            builder.append("&");
                        }
                        builder.append(encoding.getIdentifier());
                        builder.append(", ");
                    }
                }
                for (OffloadDataEncoding encoding : ((ReductionCallMapping) call).getOutputDataEncodings()) {
                    builder.append("&");

                    builder.append(encoding.getIdentifier());
                    builder.append(", ");
                }

                builder.deleteCharAt(builder.length() - 2);
                builder.append("] () {\n");
                addIndent();
                return;
            }
        }
            for (int i = 1; i < ((AssignmentExpression) call.getDefinition().getExpression()).getRhsExpression().getOperands().size(); i++) {
                Data parameter = ((AssignmentExpression) call.getDefinition().getExpression()).getRhsExpression().getOperands().get(i);
                if (!(parameter instanceof LiteralData)) {
                    if (!parameter.getIdentifier().equals(((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement().getIdentifier())) {
                        if (parameter instanceof PrimitiveData) {
                            builder.append("&");
                        }
                        builder.append(parameter.getIdentifier());
                        builder.append(", ");
                    }
                }
            }
            builder.append("&");
            builder.append(((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement().getIdentifier());
        */
        builder.append("] () {\n");
        addIndent();
    }

    /**
     * Generates the lambda header for parallel reduction call nodes on the GPU with a necessary temp variable.
     * @param lambda
     * @param call
     */
    private void createLambdaHeaderTempReduce(String lambda, ReductionCallMapping call, String tempName, String lockName) {
        indent();
        builder.append("auto ");
        builder.append(lambda);
        builder.append(" = [");
        builder.append("&");
        /*
        // create header for GPU calls
        if (call.getOnGPU()) {
            for (OffloadDataEncoding encoding : call.getInputDataEncodings()) {
                if (!(encoding.getData() instanceof LiteralData)) {
                    if (encoding.getData() instanceof PrimitiveData) {
                        builder.append("&");
                    }
                    builder.append(encoding.getIdentifier());
                    builder.append(", ");
                }
            }
        } else {
            for (int i = 1; i < ((AssignmentExpression) call.getDefinition().getExpression()).getRhsExpression().getOperands().size(); i++) {
                Data parameter = ((AssignmentExpression) call.getDefinition().getExpression()).getRhsExpression().getOperands().get(i);

                if (parameter instanceof PrimitiveData) {
                    builder.append("&");
                }
                builder.append(parameter.getIdentifier());
                builder.append(", ");

            }
        }
        builder.append("&");
        builder.append(tempName);
        builder.append(", ");
        builder.append("&");
        builder.append(lockName);

         */
        builder.append("] () {\n");
        addIndent();
    }

    /**
     * Generates the lambda header for parallel call nodes on the CPU.
     * @param lambda
     * @param call
     */
    private void createLambdaHeaderDP(String lambda, ParallelCallMapping call) {
        indent();
        builder.append("auto ");
        builder.append(lambda);
        builder.append(" = [");
        builder.append("&");

        /*
        // create header for GPU calls
        if (call instanceof GPUParallelCallMapping) {
            for (OffloadDataEncoding encoding : ((GPUParallelCallMapping) call).getInputDataEncodings()) {
                if (encoding.getData() instanceof PrimitiveData) {
                    builder.append("&");
                }
                builder.append(encoding.getIdentifier());
                builder.append(", ");
            }
            for (OffloadDataEncoding encoding : ((GPUParallelCallMapping) call).getOutputDataEncodings()) {
                builder.append("&");
                builder.append(encoding.getIdentifier());
                builder.append(", ");
            }
        } else {
            for (int i = 1; i < ((AssignmentExpression) call.getDefinition().getExpression()).getRhsExpression().getOperands().size(); i++) {
                Data parameter = ((AssignmentExpression) call.getDefinition().getExpression()).getRhsExpression().getOperands().get(i);

                builder.append(parameter.getIdentifier());
                builder.append(", ");

            }
            builder.append(((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement().getIdentifier());
            builder.append(", ");
        }

        builder.append("INDEX0_");
        builder.append(call.getCallExpression().getInlineEnding());

         */
        builder.append("] () mutable {\n");
        addIndent();
    }

    /**
     * Generates the lambda header for fused parallel call nodes on the CPU.
     * @param lambda
     * @param call
     */
    private void createFusedLambdaHeader(String lambda, FusedParallelCallMapping call) {
        indent();
        builder.append("auto ");
        builder.append(lambda);
        builder.append(" = [");
        builder.append("&");

        /*
        HashSet<Data> parameters = new HashSet<>();
        for (int i = 0; i < call.getChildren().size(); i++) {
            ParallelCallMapping callMapping = (ParallelCallMapping) call.getChildren().get(i);
            for (int j = 1; j < ((AssignmentExpression) callMapping.getDefinition().getExpression()).getRhsExpression().getOperands().size(); j++) {
                parameters.add(((AssignmentExpression) callMapping.getDefinition().getExpression()).getRhsExpression().getOperands().get(j));
            }
            parameters.add(((AssignmentExpression) callMapping.getDefinition().getExpression()).getOutputElement());
        }

        for (Data data:parameters ) {
            if (data instanceof PrimitiveData) {
                builder.append("&");
            }
            builder.append(data.getIdentifier());
            builder.append(", ");
        }
        builder.deleteCharAt(builder.length() - 2);

         */
        builder.append("] () mutable {\n");
        addIndent();
    }

    /**
     * Generates the lambda ending for parallel call nodes on the CPU.
     * @param lambda
     * @param call
     */
    private void createLambdaEnd(String lambda, ParallelCallMapping call, int offset) {
        removeIndent();
        indent();
        builder.append("};\n");
        indent();
        builder.append("getPool()->at(");
        builder.append(call.getExecutor().getRank() * call.getExecutor().getCores() + offset - 1);
        builder.append(")->addWork(");
        builder.append(lambda);
        builder.append(");\n");
    }

    /**
     * Generates the lambda ending for parallel call nodes on the CPU.
     * @param lambda
     * @param rank
     */
    private void createGPULambdaEnd(String lambda, int rank) {
        removeIndent();
        indent();
        builder.append("};\n");
        indent();
        builder.append("getGPUPool()->at(");
        builder.append(rank);
        builder.append(").addWork(");
        builder.append(lambda);
        builder.append(");\n");
    }

    /**
     * Generates the lambda ending for fused parallel call nodes on the CPU.
     * @param call
     */
    private void createFusedLambdaEnd(FusedParallelCallMapping call) {
        ParallelCallMapping callMapping = (ParallelCallMapping) call.getChildren().get(0);
        removeIndent();
        indent();
        builder.append("};\n");
        indent();
        builder.append("getPool()->at(");
        builder.append(callMapping.getExecutor().getRank() * callMapping.getExecutor().getCores() - 1);
        builder.append(").addWork(");
        builder.append("f_" + fusedLambdaExtension);
        builder.append(");\n");
    }

    /**
     * Generates a parallel version of the map pattern.
     * @param call
     * @param function
     */
    private void generateMapParallel(ParallelCallMapping call, MapMapping function, boolean doLambda) {
        // length and start are defined as the first and second additional arguments by the language standard.
        long start = call.getStartIndex().get(0);
        long length = call.getNumIterations().get(0);
        long end = start + length;
        if (needsMPI) {
            indent();
            builder.append("if (rank == ");
            builder.append(call.getExecutor().getParent().getParent().getRank());
            builder.append(") {\n");
            addIndent();
        }
        fillParallelVariableInliningTable(call,function);

        length = (long) length/call.getExecutor().getCores();

        if (call instanceof GPUParallelCallMapping) {
            String lambda = "f_gpu_" + call.getCallExpression().getInlineEnding();
            String wrapper_name = "cuda_wrapper_" + call.getFunctionIdentifier() + "_" + call.getCallExpression().getInlineEnding();
            length = call.getNumIterations().get(0);
            StringBuilder wrapper_header_builder = new StringBuilder();
            wrapper_header_builder.append("void ");
            wrapper_header_builder.append(wrapper_name);
            wrapper_header_builder.append("(");

            createLambdaHeader(lambda, call);

            indent();
            builder.append(wrapper_name);
            builder.append("(");

            ArrayList<OffloadDataEncoding> input = ((GPUParallelCallMapping) call).getInputDataEncodings();
            ArrayList<OffloadDataEncoding> output = ((GPUParallelCallMapping) call).getOutputDataEncodings();
            // get parameters
            for (OffloadDataEncoding value :  input) {
                if (((GPUParallelCallMapping) call).getOutputDataEncodings().stream().map(x -> x.getData()).allMatch(x -> x != value.getData())) {
                    Data parameter = value.getData();
                    wrapper_header_builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                    wrapper_header_builder.append("* ");
                    wrapper_header_builder.append(parameter.getIdentifier());
                    wrapper_header_builder.append(", ");

                    builder.append(value.getIdentifier());
                    builder.append(", ");
                }
            }
            for (OffloadDataEncoding value :  output) {
                Data parameter = value.getData();
                wrapper_header_builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                wrapper_header_builder.append("* ");
                wrapper_header_builder.append(parameter.getIdentifier());
                wrapper_header_builder.append(", ");
                builder.append(value.getIdentifier());
                builder.append(", ");
            }
            wrapper_header_builder.deleteCharAt(wrapper_header_builder.length() - 2);
            wrapper_header_builder.append(")");
            builder.deleteCharAt(builder.length() - 2);
            builder.append(");\n");

            createGPULambdaEnd(lambda, call.getExecutor().getParent().getGPUrank());

            String wrapper_header = wrapper_header_builder.toString();

            gpuHeaderBuilder.append(wrapper_header);
            gpuHeaderBuilder.append(";\n\n");

            int currentIndent = numIndents;
            numIndents = 0;
            gpuBuilder.append(wrapper_header);
            gpuBuilder.append(" {\n");
            addIndent();

            gpuBuilder.append("\n");


            // compute the start of halo data to ensure the correct dependencies between in-/output data elements.
            long haloStart = Long.MAX_VALUE;
            for (OffloadDataEncoding placement : input) {
                if (placement.getStart() < haloStart) {
                    haloStart = placement.getStart();
                }
            }
            for (OffloadDataEncoding placement : output) {
                if (placement.getStart() < haloStart) {
                    haloStart = placement.getStart();
                }
            }

            String kernelName = "kernel_" + wrapper_name;
            StringBuilder kernelHeaderBuilder = new StringBuilder("__global__ \nvoid " + kernelName);

            gpuIndent();
            gpuBuilder.append(kernelName);
            gpuBuilder.append("<<<");
            gpuBuilder.append(((GPUParallelCallMapping) call).getNumBlocks());
            gpuBuilder.append(", ");
            gpuBuilder.append(((GPUParallelCallMapping) call).getThreadsPerBlock());
            gpuBuilder.append(">>> (");
            kernelHeaderBuilder.append("(");

            for (OffloadDataEncoding placement : input) {
                if (((GPUParallelCallMapping) call).getOutputDataEncodings().stream().map(x -> x.getData()).allMatch(x -> x != placement.getData())) {
                    Data data = placement.getData();
                    kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                    kernelHeaderBuilder.append("* ");
                    kernelHeaderBuilder.append(" ");
                    kernelHeaderBuilder.append(placement.getData().getIdentifier());
                    kernelHeaderBuilder.append(", ");

                    gpuBuilder.append(placement.getData().getIdentifier());
                    gpuBuilder.append(", ");
                }
            }

            for (OffloadDataEncoding placement : output) {
                Data data = placement.getData();
                kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                kernelHeaderBuilder.append("*");
                kernelHeaderBuilder.append(" ");
                kernelHeaderBuilder.append(placement.getData().getIdentifier());
                kernelHeaderBuilder.append(", ");

                gpuBuilder.append(placement.getData().getIdentifier());
                gpuBuilder.append(", ");
            }

            gpuBuilder.deleteCharAt(gpuBuilder.length() - 2);
            kernelHeaderBuilder.deleteCharAt(kernelHeaderBuilder.length() - 2);

            kernelHeaderBuilder.append(")");
            gpuBuilder.append(");\n");

            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n\n");


            gpuKernelHeaderBuilder.append(kernelHeaderBuilder);
            gpuKernelHeaderBuilder.append(";\n\n");

            gpuBuilder.append(kernelHeaderBuilder);

            gpuBuilder.append(" {\n");
            addIndent();
            gpuIndent();

            gpuBuilder.append("int tid = blockIdx.x * blockDim.x + threadIdx.x;\n");

            // Define a cutoff for increasing the workload for a subset of threads on the gpu in order to compute the exact pattern.
            int range = (int) (length / (((GPUParallelCallMapping) call).getNumBlocks() * ((GPUParallelCallMapping) call).getThreadsPerBlock()));
            int cutoff = range * (((GPUParallelCallMapping) call).getNumBlocks() * ((GPUParallelCallMapping) call).getThreadsPerBlock());
            boolean useCutoff = cutoff < length;

            // Internal computation
            gpuIndent();
            gpuBuilder.append("int exec_range = ");
            gpuBuilder.append(range);
            gpuBuilder.append(";\n");

            if (useCutoff) {
                gpuIndent();
                gpuBuilder.append("if (tid < ");
                gpuBuilder.append(length - cutoff);
                gpuBuilder.append(") {\n");
                addIndent();
                gpuIndent();
                gpuBuilder.append("exec_range++;\n");
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
            }

            gpuIndent();
            gpuBuilder.append("for ( int range_iterator = ");
            gpuBuilder.append(start - haloStart);
            gpuBuilder.append("; range_iterator < exec_range + ");
            gpuBuilder.append(start - haloStart);
            gpuBuilder.append("; range_iterator++) {\n");
            addIndent();
            gpuIndent();
            String indexName = "INDEX_" + call.getCallExpression().getInlineEnding();
            gpuBuilder.append("int ");
            gpuBuilder.append(indexName);
            gpuBuilder.append(" = tid * exec_range + range_iterator");
            if (useCutoff) {
                gpuBuilder.append(" + ");
                gpuBuilder.append(length - cutoff);
            }
            gpuBuilder.append(";\n");
            if (useCutoff) {
                gpuIndent();
                gpuBuilder.append("if (tid < ");
                gpuBuilder.append(length - cutoff);
                gpuBuilder.append(") {\n");
                addIndent();
                gpuIndent();
                gpuBuilder.append(indexName);
                gpuBuilder.append(" -= ");
                gpuBuilder.append(length - cutoff);
                gpuBuilder.append(";\n");
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
            }

            // Update name
            for (OffloadDataEncoding placement : input) {
                if (placement.getData() instanceof PrimitiveData) {
                    placement.updateDataIdentifier();
                }
            }

            // create gpu context
            onGPU = true;

            StringBuilder cpuBuilder = builder;
            builder = gpuBuilder;

            addVariables(function);

            for (MappingNode child : function.getChildren()) {
                child.accept(getRealThis());
            }

            closeScope(function);
            gpuBuilder = builder;
            builder = cpuBuilder;
            onGPU = false;
            // end gpu context
            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n");

            // Reset name
            for (OffloadDataEncoding placement: input ) {
                if (placement.getData() instanceof PrimitiveData) {
                    placement.resetDataIdentifier();
                }
            }
            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n\n");

            numIndents = currentIndent;
        } else {
            int core_offset = 0;
            if (call.getExecutor().isFirstCG()) {
                length = (length*call.getExecutor().getCores())/(call.getExecutor().getCores() - 1);
                core_offset = 1;
            }
            for (int i = core_offset; i < call.getExecutor().getCores(); i++) {
                String lambda = "f_" + i + "_" + call.getCallExpression().getInlineEnding();
                if (doLambda) {
                    createLambdaHeader(lambda, call);
                }
                // length and start are defined as the first and second additional arguments by the language standard.
                String index = "INDEX_" + call.getCallExpression().getInlineEnding();

                if (i == call.getExecutor().getCores() - 1) {
                    length = end - start;
                }

                indent();
                builder.append("for (size_t ");
                builder.append(index);
                builder.append(" = ");
                builder.append(start);
                builder.append("; ");
                builder.append(index);
                builder.append(" < ");
                builder.append(start);
                builder.append(" + ");
                builder.append(length);
                builder.append("; ++");
                builder.append(index);
                builder.append(") {\n");

                start = start + length;
                addIndent();

                addVariables(function);

                for (MappingNode child : function.getChildren()) {
                    child.accept(getRealThis());
                }

                closeScope(function);

                removeIndent();
                indent();
                builder.append("}\n");
                if (doLambda) {
                    createLambdaEnd(lambda, call, i);
                }

            }
        }
        if (needsMPI) {
            removeIndent();
            indent();
            builder.append("}\n");
        }
        clearParallelVariableInliningTable(function);
    }

    /**
     * Returns the neutral element for a given combiner function and a data type.
     * @param combinerFunction
     * @param dataType
     * @return
     */
    private String getNeutralElement(CombinerFunction combinerFunction, PrimitiveDataTypes dataType) {
        StringBuilder subBuilder = new StringBuilder();

        subBuilder.append("std::numeric_limits<");
        subBuilder.append(CppTypesPrinter.doPrintType(dataType));
        subBuilder.append(">::");

        if (combinerFunction == CombinerFunction.PLUS) {
            return "0";
        } else if (combinerFunction == CombinerFunction.TIMES) {
            return "1";
        } else if (combinerFunction == CombinerFunction.MAX) {
            subBuilder.append("max()");
            return String.valueOf(PrimitiveDataTypes.getMinValue(dataType));
            //return subBuilder.toString();
        } else if (combinerFunction == CombinerFunction.MIN) {
            subBuilder.append("min()");
            return String.valueOf(PrimitiveDataTypes.getMaxValue(dataType));
            //return subBuilder.toString();
        }
        return "";
    }

    /**
     * Generates a parallel version of the reduction pattern.
     * @param call
     * @param function
     */
    private void generateReduceParallel(ReductionCallMapping call, ReduceMapping function, boolean doLambda) {

        String reductionLock = "reduction_lock_" + call.getGroup().getGroupIdentifier();
        String tempDataName = "temp_data_" + call.getGroup().getGroupIdentifier();
        long length = call.getNumIterations().get(0);
        long start = call.getStartIndex().get(0);

        boolean hasTempData = false;

        if (!call.getTempOutput().isEmpty()) {
            hasTempData = true;
        }

        // generate MPI applicable Temp data
        if (call.getGroup().isFirstAccess() && !call.getTempOutput().isEmpty()) {
            call.getGroup().setFirstAccess(false);
            indent();
            builder.append("pthread_mutex_t ");
            builder.append(reductionLock);
            builder.append(" = PTHREAD_MUTEX_INITIALIZER;\n");
            indent();
            builder.append(CppTypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
            builder.append(" ");
            builder.append(tempDataName);
            builder.append(" = ");
            builder.append(getNeutralElement(function.getCombinerFunction(), function.getReturnElement().getTypeName()));
            builder.append(";\n");

        }

        if (call.isOnlyCombiner()) {
            HashSet<Node> nodes = new HashSet<>();
            for (Processor processor: call.getGroup().getProcessors() ) {
                nodes.add(processor.getParent().getParent());
            }

            if (nodes.size() > 1) {
                indent();
                StringBuilder resultBuilder = new StringBuilder();
                resultBuilder.append(CppExpressionPrinter.doPrintExpression(call.getDefinition().getExpression(), true, aktiveFunctions, activePatterns));

                resultBuilder.delete(resultBuilder.indexOf(" ="), resultBuilder.length());

                builder.append(CppTypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
                String mpi_combiner = "MPI_Reduction_Combiner_" + RandomStringGenerator.getAlphaNumericString();
                builder.append(" ");
                builder.append(mpi_combiner);
                builder.append(" = ");
                builder.append(getNeutralElement(function.getCombinerFunction(), function.getReturnElement().getTypeName()));
                builder.append(";\n");

                indent();
                builder.append("MPI_Reduce(&");
                builder.append(tempDataName);
                builder.append(", &");

                builder.append(mpi_combiner);
                builder.append(", 1, ");
                builder.append(MPITypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
                builder.append(", ");
                if (function.getCombinerFunction() == CombinerFunction.MAX) {
                    builder.append("MPI_MAX, ");
                } else if (function.getCombinerFunction() == CombinerFunction.MIN) {
                    builder.append("MPI_MIN, ");
                } else if (function.getCombinerFunction() == CombinerFunction.PLUS) {
                    builder.append("MPI_SUM, ");
                }else if (function.getCombinerFunction() == CombinerFunction.TIMES) {
                    builder.append("MPI_PROD, ");
                }
                builder.append(call.getTargetNode().getRank());
                builder.append(", MPI_COMM_WORLD);\n");
                indent();
                builder.append("if (rank == ");
                builder.append(call.getTargetNode().getRank());
                builder.append(") {\n");
                addIndent();
                indent();
                builder.append(resultBuilder);
                builder.append(" = ");
                builder.append(mpi_combiner);
                builder.append(";\n");
                removeIndent();
                indent();
                builder.append("}\n");
            } else {
                indent();
                StringBuilder resultBuilder = new StringBuilder();
                resultBuilder.append(CppExpressionPrinter.doPrintExpression(call.getDefinition().getExpression(), true, aktiveFunctions, activePatterns));

                resultBuilder.delete(resultBuilder.indexOf(" ="), resultBuilder.length());

                builder.append(resultBuilder);
                builder.append(" = ");
                builder.append(tempDataName);
                builder.append(";\n");
            }
        } else if (call.getOnGPU()) {
            if (needsMPI) {
                indent();
                builder.append("if (rank == ");
                builder.append(call.getExecutor().getParent().getParent().getRank());
                builder.append(") {\n");
                addIndent();
            }
            fillParallelVariableInliningTable(call,function);
            String lambda = "f_gpu_" + call.getCallExpression().getInlineEnding();
            String wrapper_name = "cuda_wrapper_" + call.getFunctionIdentifier() + "_" + call.getCallExpression().getInlineEnding();
            length = call.getNumIterations().get(0);
            StringBuilder wrapper_header_builder = new StringBuilder();
            wrapper_header_builder.append("void ");
            wrapper_header_builder.append(wrapper_name);
            wrapper_header_builder.append("(");

            if (hasTempData) {
                createLambdaHeaderTempReduce(lambda, call, tempDataName, reductionLock);
            } else {
                createLambdaHeader(lambda, call);
            }

            ArrayList<OffloadDataEncoding> input = call.getInputDataEncodings();
            ArrayList<OffloadDataEncoding> output = call.getOutputDataEncodings();

            indent();
            builder.append(wrapper_name);
            builder.append("(");

            // get parameters
            for (OffloadDataEncoding value :  input) {
                if ( call.getOutputDataEncodings().stream().map(x -> x.getData()).allMatch(x -> x != value.getData())) {
                    Data parameter = value.getData();
                    wrapper_header_builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                    wrapper_header_builder.append("* ");
                    wrapper_header_builder.append(parameter.getIdentifier());
                    wrapper_header_builder.append(", ");
                    builder.append(value.getIdentifier());
                    builder.append(", ");
                }
            }
            if (hasTempData) {
                Data outputElement = ((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement();
                wrapper_header_builder.append(CppTypesPrinter.doPrintType(outputElement.getTypeName()));
                wrapper_header_builder.append("*");
                wrapper_header_builder.append(" ");
                wrapper_header_builder.append(tempDataName);
                wrapper_header_builder.append(", pthread_mutex_t ");
                wrapper_header_builder.append(reductionLock);
                wrapper_header_builder.append(")");

                builder.append("&");
                builder.append(tempDataName);
                builder.append(", ");
                builder.append(reductionLock);

            } else {
                for (OffloadDataEncoding value : output) {
                    Data parameter = value.getData();
                    wrapper_header_builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                    wrapper_header_builder.append("* ");
                    wrapper_header_builder.append(parameter.getIdentifier());
                    wrapper_header_builder.append(", ");
                    builder.append(value.getIdentifier());
                    builder.append(", ");
                }
                wrapper_header_builder.deleteCharAt(wrapper_header_builder.length() - 2);
                wrapper_header_builder.append(")");
                builder.deleteCharAt(builder.length() - 2);
            }
            builder.append(");\n");

            createGPULambdaEnd(lambda, call.getExecutor().getParent().getGPUrank());

            String wrapper_header = wrapper_header_builder.toString();

            gpuHeaderBuilder.append(wrapper_header);
            gpuHeaderBuilder.append(";\n\n");

            int currentIndent = numIndents;
            numIndents = 0;
            gpuBuilder.append(wrapper_header);
            gpuBuilder.append(" {\n");
            addIndent();

            String tempRes = "temp_res";

            if (hasTempData) {
                gpuIndent();
                gpuBuilder.append(CppTypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
                gpuBuilder.append("* ");
                gpuBuilder.append(tempRes);
                gpuBuilder.append(";\n");
                gpuIndent();
                gpuBuilder.append(CppTypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
                gpuBuilder.append("* d_");
                gpuBuilder.append(tempRes);
                gpuBuilder.append(";\n");
                gpuIndent();
                gpuBuilder.append(tempRes);
                gpuBuilder.append(" = Init_List(");
                gpuBuilder.append(tempRes);
                gpuBuilder.append(", 1);\n");
                gpuIndent();
                gpuBuilder.append("cudaMalloc(&d_");
                gpuBuilder.append(tempRes);
                gpuBuilder.append(", 1 * sizeof(");
                gpuBuilder.append(CppTypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
                gpuBuilder.append("));\n");

            }


            gpuBuilder.append("\n");

            // compute the start of halo data to ensure the correct dependencies between input data elements.
            long haloStart = Long.MAX_VALUE;
            for (OffloadDataEncoding placement : input) {
                if (placement.getStart() < haloStart) {
                    haloStart = placement.getStart();
                }
            }

            String kernelName = "kernel_" + wrapper_name;
            StringBuilder kernelHeaderBuilder = new StringBuilder("__global__ \nvoid " + kernelName);

            gpuIndent();
            gpuBuilder.append(kernelName);
            gpuBuilder.append("<<<");
            gpuBuilder.append(call.getNumBlocks());
            gpuBuilder.append(", ");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(">>> (");
            kernelHeaderBuilder.append("(");
            for (OffloadDataEncoding placement : input) {
                if (call.getOutputDataEncodings().stream().map(x -> x.getData()).allMatch(x -> x != placement.getData())) {
                    Data data = placement.getData();
                    kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                    kernelHeaderBuilder.append("*");
                    kernelHeaderBuilder.append(" ");
                    kernelHeaderBuilder.append(data.getIdentifier());
                    kernelHeaderBuilder.append(", ");

                    gpuBuilder.append(data.getIdentifier());

                    gpuBuilder.append(", ");
                }
            }

            if (hasTempData) {
                kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
                kernelHeaderBuilder.append("* ");
                kernelHeaderBuilder.append(call.getOutputElements().get(0).getIdentifier());
                kernelHeaderBuilder.append(", ");

                gpuBuilder.append("d_");
                gpuBuilder.append(tempRes);
                gpuBuilder.append(", ");
            } else {
                for (OffloadDataEncoding placement : output) {
                    Data data = placement.getData();
                    kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                    kernelHeaderBuilder.append("* ");
                    kernelHeaderBuilder.append(data.getIdentifier());
                    kernelHeaderBuilder.append(", ");

                    gpuBuilder.append(data.getIdentifier());
                    gpuBuilder.append(", ");
                }
            }



            gpuBuilder.deleteCharAt(gpuBuilder.length() - 2);
            kernelHeaderBuilder.deleteCharAt(kernelHeaderBuilder.length() - 2);

            kernelHeaderBuilder.append(")");
            gpuBuilder.append(");\n");

            // generate reduction over multiple blocks
            if (call.getNumBlocks() > 1) {
                gpuIndent();
                String blockCombiner = "FAILURE";
                CombinerFunction combiner = function.getCombinerFunction();
                if (combiner == CombinerFunction.MAX) {
                    blockCombiner = "cuda_reduce_max";
                } else if (combiner == CombinerFunction.MIN) {
                    blockCombiner = "cuda_reduce_min";
                } else if (combiner == CombinerFunction.PLUS) {
                    blockCombiner = "cuda_reduce_sum";
                } else if (combiner == CombinerFunction.TIMES) {
                    blockCombiner = "cuda_reduce_times";
                }
                gpuBuilder.append(blockCombiner);
                gpuBuilder.append("<<<1, ");
                gpuBuilder.append(call.getNumBlocks());
                gpuBuilder.append(">>> (");
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append(", ");
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append(", ");
                gpuBuilder.append(call.getNumBlocks());
                gpuBuilder.append(");\n\n");
            }


            if (hasTempData) {

                gpuIndent();
                gpuBuilder.append("cudaMemcpy(&");
                gpuBuilder.append(tempRes);
                gpuBuilder.append("[0], &d_");
                gpuBuilder.append(tempRes);
                gpuBuilder.append("[0], sizeof(");
                gpuBuilder.append(CppTypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
                gpuBuilder.append(") * 1, cudaMemcpyDeviceToHost);\n");
                gpuIndent();
                gpuBuilder.append("cudaFree(d_");
                gpuBuilder.append(tempRes);
                gpuBuilder.append(");\n");

                gpuIndent();
                gpuBuilder.append("pthread_mutex_lock(&");
                gpuBuilder.append(reductionLock);
                gpuBuilder.append(");\n");

                gpuIndent();
                gpuBuilder.append(tempDataName);
                gpuBuilder.append("[0] = ");
                CombinerFunction combiner = function.getCombinerFunction();
                if (combiner == CombinerFunction.MAX) {
                    gpuBuilder.append("reduction_max(");
                    gpuBuilder.append(tempDataName);
                    gpuBuilder.append("[0], ");
                    gpuBuilder.append(tempRes);
                    gpuBuilder.append(", 1, 0);\n");
                } else if (combiner == CombinerFunction.MIN) {
                    gpuBuilder.append("reduction_min(");
                    gpuBuilder.append(tempDataName);
                    gpuBuilder.append("[0], ");
                    gpuBuilder.append(tempRes);
                    gpuBuilder.append(", 1, 0);\n");
                } else if (combiner == CombinerFunction.PLUS) {
                    gpuBuilder.append("reduction_sum(");
                    gpuBuilder.append(tempDataName);
                    gpuBuilder.append("[0], ");
                    gpuBuilder.append(tempRes);
                    gpuBuilder.append(", 1, 0);\n");
                } else if (combiner == CombinerFunction.TIMES) {
                    gpuBuilder.append("reduction_times(");
                    gpuBuilder.append(tempDataName);
                    gpuBuilder.append("[0], ");
                    gpuBuilder.append(tempRes);
                    gpuBuilder.append(", 1, 0);\n");
                }

                gpuIndent();
                gpuBuilder.append("pthread_mutex_unlock(&");
                gpuBuilder.append(reductionLock);
                gpuBuilder.append(");\n");
            }
            gpuIndent();
            if (hasTempData) {
                gpuBuilder.append("std::free(");
                gpuBuilder.append(tempRes);
                gpuBuilder.append(");\n");
            }
            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n\n");


            gpuBuilder.append(kernelHeaderBuilder);

            gpuKernelHeaderBuilder.append(kernelHeaderBuilder);
            gpuKernelHeaderBuilder.append(";\n\n");

            gpuBuilder.append(" {\n");
            addIndent();
            gpuIndent();

            gpuBuilder.append("int tid = blockIdx.x * blockDim.x + threadIdx.x;\n");
            gpuIndent();
            gpuBuilder.append("int id = threadIdx.x;\n");
            gpuIndent();
            gpuBuilder.append("__shared__ ");
            gpuBuilder.append(CppTypesPrinter.doPrintType(call.getOutputElements().get(0).getTypeName()));
            gpuBuilder.append(" ");
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s [");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append("];\n");

            gpuIndent();
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id] = ");
            gpuBuilder.append(getNeutralElement(function.getCombinerFunction(), call.getOutputElements().get(0).getTypeName()));
            gpuBuilder.append(";\n\n");

            // Define a cutoff for increasing the workload for a subset of threads on the gpu in order to compute the exact pattern.
            int range = (int) (length / (call.getNumBlocks() * call.getNumThreads()));
            int cutoff = range * ( call.getNumBlocks() * call.getNumThreads());
            boolean useCutoff = cutoff < length;

            gpuIndent();
            gpuBuilder.append("int exec_range = ");
            gpuBuilder.append(range);
            gpuBuilder.append(";\n");

            if (useCutoff) {
                gpuIndent();
                gpuBuilder.append("if (tid < ");
                gpuBuilder.append(length - cutoff);
                gpuBuilder.append(") {\n");
                addIndent();
                gpuIndent();
                gpuBuilder.append("exec_range++;\n");
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
            }

            gpuIndent();
            gpuBuilder.append("for ( size_t range_iterator = ");
            gpuBuilder.append(start - haloStart);
            gpuBuilder.append("; range_iterator < exec_range + ");
            gpuBuilder.append(start - haloStart);
            gpuBuilder.append("; range_iterator++) {\n");
            addIndent();
            gpuIndent();
            String indexName = "INDEX_" + call.getCallExpression().getInlineEnding();
            gpuBuilder.append("int ");
            gpuBuilder.append(indexName);
            gpuBuilder.append(" = tid * exec_range + range_iterator");
            if (useCutoff) {
                gpuBuilder.append(" + ");
                gpuBuilder.append(length - cutoff);
            }
            gpuBuilder.append(";\n");
            if (useCutoff) {
                gpuIndent();
                gpuBuilder.append("if (tid < ");
                gpuBuilder.append(length - cutoff);
                gpuBuilder.append(") {\n");
                addIndent();
                gpuIndent();
                gpuBuilder.append(indexName);
                gpuBuilder.append(" -= ");
                gpuBuilder.append(length - cutoff);
                gpuBuilder.append(";\n");
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
            }


            for (OffloadDataEncoding placement : input) {
                placement.updateDataIdentifier();
            }

            for (OffloadDataEncoding placement : output) {
                placement.updateDataIdentifier(true);
            }

            // create gpu context
            onGPU = true;

            StringBuilder cpuBuilder = builder;
            builder = gpuBuilder;

            addVariables(function);

            for (MappingNode child : function.getChildren()) {
                child.accept(getRealThis());
            }

            closeScope(function);
            gpuBuilder = builder;
            builder = cpuBuilder;
            onGPU = false;
            // end gpu context
            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n");
            for (OffloadDataEncoding placement: input ) {
                placement.resetDataIdentifier();
            }
            for (OffloadDataEncoding placement: output ) {
                placement.resetDataIdentifier(true);
            }

            gpuIndent();
            gpuBuilder.append("__syncthreads();\n\n");

            String combinerStart = "";
            String combinerMiddle = "";
            String combinerEnd = "";

            CombinerFunction combiner = function.getCombinerFunction();
            if (combiner == CombinerFunction.MAX) {

                combinerStart = "max(";
                combinerMiddle = ", ";
                combinerEnd = ")";
            } else if (combiner == CombinerFunction.MIN) {
                combinerStart = "min(";
                combinerMiddle = ", ";
                combinerEnd = ")";
            } else if (combiner == CombinerFunction.PLUS) {
                combinerMiddle = " += ";
            } else if (combiner == CombinerFunction.TIMES) {
                combinerMiddle = " *= ";
            }

            // generate block local reduction
            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 512) { if (id < 256) { ");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 256]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; } __syncthreads(); }\n");

            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 256) { if (id < 128) { ");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 128]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; } __syncthreads(); }\n");

            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 128) { if (id < 64) { ");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 64]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; } __syncthreads(); }\n\n");

            gpuIndent();
            gpuBuilder.append("if ( id < 32) {\n");
            addIndent();

            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 64) {");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 32]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; }\n");
            gpuIndent();
            gpuBuilder.append("__syncwarp();\n");

            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 32) {");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 16]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; }\n");
            gpuIndent();
            gpuBuilder.append("__syncwarp();\n");

            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 16) {");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 8]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; }\n");
            gpuIndent();
            gpuBuilder.append("__syncwarp();\n");

            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 8) {");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 4]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; }\n");
            gpuIndent();
            gpuBuilder.append("__syncwarp();\n");

            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 4) {");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 2]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; }\n");
            gpuIndent();
            gpuBuilder.append("__syncwarp();\n");

            gpuIndent();
            gpuBuilder.append("if (");
            gpuBuilder.append(call.getNumThreads());
            gpuBuilder.append(" >= 2) {");
            if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[id] = ");
            }
            gpuBuilder.append(combinerStart);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id]");
            gpuBuilder.append(combinerMiddle);
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[id + 1]");
            gpuBuilder.append(combinerEnd);
            gpuBuilder.append("; }\n");

            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n");
            gpuIndent();
            gpuBuilder.append("__syncwarp();\n");

            gpuIndent();
            gpuBuilder.append("if ( id == 0) {\n");
            addIndent();


            // handle remaining data
            long remainderStart = 0;
            boolean doRemainder = false;
            for (long i = 1; i < 9; i++) {
                if (call.getNumThreads() < Math.pow(2, i + 1) && Math.pow(2, i) < call.getNumThreads()) {
                    remainderStart = (long) Math.pow(2, i);
                    doRemainder = true;
                    break;
                }
            }
            if (doRemainder) {
                gpuIndent();
                gpuBuilder.append("if ( tid == 0) {\n");
                addIndent();
                gpuIndent();
                gpuBuilder.append("for (size_t i = ");
                gpuBuilder.append(remainderStart);
                gpuBuilder.append("; i < ");
                gpuBuilder.append(call.getNumThreads());
                gpuBuilder.append("; i++) {\n");
                addIndent();
                gpuIndent();
                if (combiner == CombinerFunction.MAX || combiner == CombinerFunction.MIN) {
                    gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                    gpuBuilder.append("_s[0] = ");
                }
                gpuBuilder.append(combinerStart);
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[0]");
                gpuBuilder.append(combinerMiddle);
                gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
                gpuBuilder.append("_s[i]");
                gpuBuilder.append(combinerEnd);
                gpuBuilder.append(";\n");
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
            }

            gpuIndent();
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("[blockIdx.x] = ");
            gpuBuilder.append(call.getOutputElements().get(0).getIdentifier());
            gpuBuilder.append("_s[0];\n");

            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n");

            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n\n");

            numIndents = currentIndent;
            if (needsMPI) {
                removeIndent();
                indent();
                builder.append("}\n");
            }
            clearParallelVariableInliningTable(function);
        } else  {
            // length and start are defined as the first and second additional arguments by the language standard.
            length = call.getNumIterations().get(0);
            long end = start + length;





            if (needsMPI) {
                indent();
                builder.append("if (rank == ");
                builder.append(call.getExecutor().getParent().getParent().getRank());
                builder.append(") {\n");
                addIndent();
            }
            indent();
            String partialName = "partialResult_" + call.getCallExpression().getInlineEnding();

            builder.append(CppTypesPrinter.doPrintType(function.getReturnElement().getTypeName()));
            builder.append("* ");
            builder.append(partialName);
            builder.append( ";\n");
            indent();

            builder.append(partialName);
            builder.append(" = Init_List(");
            builder.append(((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement().getIdentifier());
            builder.append(", ");
            builder.append(partialName);
            builder.append(", ");
            builder.append(call.getExecutor().getCores());
            builder.append(");\n");

            fillParallelVariableInliningTable(call,function);

            length = (long) length/call.getExecutor().getCores();

            String oldIdentifier = ((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement().getBaseIdentifier();


            int core_offset = 0;
            if (call.getExecutor().isFirstCG()) {
                length = (length*call.getExecutor().getCores())/(call.getExecutor().getCores() - 1);
                core_offset = 1;
            }
            for (int i = core_offset; i < call.getExecutor().getCores(); i++) {
                ((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement().setIdentifier(partialName );
                String lambda = "f_" + (i - core_offset) + "_" + call.getCallExpression().getInlineEnding();
                if (doLambda) {
                    createLambdaHeader(lambda, call);
                }
                ((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement().setIdentifier(partialName +"[" + (i - core_offset) +"]");
                // length and start are defined as the first and second additional arguments by the language standard.
                String index = "INDEX_" + call.getCallExpression().getInlineEnding();

                if (i == call.getExecutor().getCores() - 1) {
                    length = end - start;
                }

                indent();
                builder.append("for (size_t ");
                builder.append(index);
                builder.append(" = ");
                builder.append(start);
                builder.append("; ");
                builder.append(index);
                builder.append(" < ");
                builder.append(start);
                builder.append(" + ");
                builder.append(length);
                builder.append("; ++");
                builder.append(index);
                builder.append(") {\n");

                start = start + length;
                addIndent();

                addVariables(function);

                for (MappingNode child : function.getChildren()) {
                    child.accept(getRealThis());
                }

                closeScope(function);

                removeIndent();
                indent();
                builder.append("}\n");
                createLambdaEnd(lambda,call,i);

            }
            ((AssignmentExpression) call.getDefinition().getExpression()).getOutputElement().setIdentifier(oldIdentifier);

            String maskName = "mask_ptr_" + call.getCallExpression().getInlineEnding();
            indent();
            builder.append("Bit_Mask * ");
            builder.append(maskName);
            builder.append(" = new Bit_Mask(");
            int numCores = call.getExecutor().getCores() * call.getExecutor().getParent().getProcessor().size() - 1;
            builder.append(numCores);
            builder.append(",false);\n");
            indent();
            String iterName = "i_" + call.getCallExpression().getInlineEnding();
            builder.append("for (size_t ");
            builder.append(iterName);
            builder.append(" = ");
            int offset = 0;
            if (call.getExecutor().isFirstCG()) {
                offset = 1;
            }
            builder.append(call.getExecutor().getCores() * call.getExecutor().getRank() - offset);
            builder.append("; ");
            builder.append(iterName);
            builder.append(" < ");
            builder.append(call.getExecutor().getCores() * (call.getExecutor().getRank() + 1) - 1);
            builder.append("; ++");
            builder.append(iterName);
            builder.append(") {\n");
            addIndent();
            indent();
            builder.append(maskName);
            builder.append("->setBarrier(");
            builder.append(iterName);
            builder.append(");\n");
            removeIndent();
            indent();
            builder.append("}\n");
            indent();
            builder.append("boost::shared_ptr<Bit_Mask>");
            String sharedMaskName = "mask_" + call.getCallExpression().getInlineEnding();
            builder.append(sharedMaskName);
            builder.append(" (");
            builder.append(maskName);
            builder.append(");\n");
            indent();
            builder.append("barrier(");
            builder.append(sharedMaskName);
            builder.append(");\n");

            CombinerFunction combiner = function.getCombinerFunction();
            indent();

            String subreductionLambda = "lambda_reduction_" + call.getCallExpression().getInlineEnding();
            builder.append("auto ");
            builder.append(subreductionLambda);
            builder.append(" = [");

            builder.append(partialName);
            builder.append(", &");
            if (hasTempData) {
                builder.append(tempDataName);
            } else {
                builder.append(call.getOutputElements().get(0).getIdentifier());
            }
            if (hasTempData) {
                builder.append(" ,&");
                builder.append(reductionLock);
            }
            builder.append("] () {\n");
            addIndent();
            if (hasTempData) {
                indent();
                builder.append("pthread_mutex_lock(&");
                builder.append(reductionLock);
                builder.append(");\n");
            }

            indent();
            if (hasTempData) {
                builder.append(tempDataName);
            } else {
                builder.append(call.getOutputElements().get(0).getIdentifier());
            }
            builder.append(" = ");
            int numResults = call.getExecutor().getCores();
            if (call.getExecutor().isFirstCG()) {
                numResults -= 1;
            }
            if (combiner == CombinerFunction.MAX) {
                builder.append("reduction_max(");
                if (hasTempData) {
                    builder.append(tempDataName);
                } else {
                    builder.append(call.getOutputElements().get(0).getIdentifier());
                }
                builder.append(", ");
                builder.append(partialName);
                builder.append(", ");
                builder.append(numResults);
                builder.append(", ");
                builder.append("0");
                builder.append(");\n");
            } else if (combiner == CombinerFunction.MIN) {
                builder.append("reduction_min(");
                if (hasTempData) {
                    builder.append(tempDataName);
                } else {
                    builder.append(call.getOutputElements().get(0).getIdentifier());
                }
                builder.append(", ");
                builder.append(partialName);
                builder.append(", ");
                builder.append(numResults);
                builder.append(", ");
                builder.append("0");
                builder.append(");\n");
            } else if (combiner == CombinerFunction.PLUS) {
                builder.append("reduction_sum(");
                if (hasTempData) {
                    builder.append(tempDataName);
                } else {
                    builder.append(call.getOutputElements().get(0).getIdentifier());
                }
                builder.append(", ");
                builder.append(partialName);
                builder.append(", ");
                builder.append(numResults);
                builder.append(", ");
                builder.append("0");
                builder.append(");\n");
            } else if (combiner == CombinerFunction.TIMES) {
                builder.append("reduction_times(");
                if (hasTempData) {
                    builder.append(tempDataName);
                } else {
                    builder.append(call.getOutputElements().get(0).getIdentifier());
                }
                builder.append(", ");
                builder.append(partialName);
                builder.append(", ");
                builder.append(numResults);
                builder.append(", ");
                builder.append("0");
                builder.append(");\n");
            }

            if (hasTempData) {
                indent();
                builder.append("pthread_mutex_unlock(&");
                builder.append(reductionLock);
                builder.append(");\n");
            }

            if (doLambda) {
                createLambdaEnd(subreductionLambda, call, 1);
            }
            if (needsMPI) {
                removeIndent();
                indent();
                builder.append("}\n");
            }
            clearParallelVariableInliningTable(function);
        }
    }

    /**
     * Generates a parallel version of the stencil pattern.
     * @param call
     * @param function
     */
    private void generateStencilParallel(ParallelCallMapping call, StencilMapping function, boolean doLambda) {
        fillParallelVariableInliningTable(call,function);

        ArrayList<Long> lengths = new ArrayList<>(call.getNumIterations());
        ArrayList<Long> starts = new ArrayList<>(call.getStartIndex());
        int dimensions = function.getDimension();
        long end = starts.get(0) + lengths.get(0);
        if (needsMPI) {
            indent();
            builder.append("if (rank == ");
            builder.append(call.getExecutor().getParent().getParent().getRank());
            builder.append(") {\n");
            addIndent();
        }
        lengths.set(0,(long) lengths.get(0)/call.getExecutor().getCores());

        if (call instanceof GPUParallelCallMapping) {
            String lambda = "f_gpu_" + call.getCallExpression().getInlineEnding();
            String wrapper_name = "cuda_wrapper_" + call.getFunctionIdentifier() + "_" + call.getCallExpression().getInlineEnding();
            lengths.set(0, call.getNumIterations().get(0));
            StringBuilder wrapper_header_builder = new StringBuilder();
            wrapper_header_builder.append("void ");
            wrapper_header_builder.append(wrapper_name);
            wrapper_header_builder.append("(");

            createLambdaHeader(lambda, call);

            indent();
            builder.append(wrapper_name);
            builder.append("(");

            ArrayList<OffloadDataEncoding> input = ((GPUParallelCallMapping) call).getInputDataEncodings();
            ArrayList<OffloadDataEncoding> output = ((GPUParallelCallMapping) call).getOutputDataEncodings();


            // get parameters
            for (OffloadDataEncoding value :  input) {
                if (((GPUParallelCallMapping) call).getOutputDataEncodings().stream().map(x -> x.getData()).allMatch(x -> x != value.getData())) {
                    Data parameter = value.getData();
                    wrapper_header_builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                    wrapper_header_builder.append("* ");
                    wrapper_header_builder.append(parameter.getIdentifier());
                    wrapper_header_builder.append(", ");
                    builder.append(value.getIdentifier());
                    builder.append(", ");
                }
            }
            for (OffloadDataEncoding value :  output) {
                Data parameter = value.getData();
                wrapper_header_builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                wrapper_header_builder.append("* ");
                wrapper_header_builder.append(parameter.getIdentifier());
                wrapper_header_builder.append(", ");
                builder.append(value.getIdentifier());
                builder.append(", ");
            }
            wrapper_header_builder.deleteCharAt(wrapper_header_builder.length() - 2);
            wrapper_header_builder.append(")");
            builder.deleteCharAt(builder.length() - 2);
            builder.append(");\n");

            createGPULambdaEnd(lambda, call.getExecutor().getParent().getGPUrank());

            String wrapper_header = wrapper_header_builder.toString();

            gpuHeaderBuilder.append(wrapper_header);
            gpuHeaderBuilder.append(";\n\n");

            int currentIndent = numIndents;
            numIndents = 0;
            gpuBuilder.append(wrapper_header);
            gpuBuilder.append(" {\n");
            addIndent();

            gpuBuilder.append("\n");

            // compute the start of halo data to ensure the correct dependencies between in-/output data elements.
            long haloStart = Long.MAX_VALUE;
            for (OffloadDataEncoding placement : input) {
                if (placement.getStart() < haloStart) {
                    haloStart = placement.getStart();
                }
            }
            for (OffloadDataEncoding placement : output) {
                if (placement.getStart() < haloStart) {
                    haloStart = placement.getStart();
                }
            }


            String kernelName = "kernel_" + wrapper_name;
            StringBuilder kernelHeaderBuilder = new StringBuilder("__global__ \nvoid " + kernelName);

            gpuIndent();
            gpuBuilder.append(kernelName);
            gpuBuilder.append("<<<");
            gpuBuilder.append(((GPUParallelCallMapping) call).getNumBlocks());
            gpuBuilder.append(", ");
            gpuBuilder.append(((GPUParallelCallMapping) call).getThreadsPerBlock());
            gpuBuilder.append(">>> (");
            kernelHeaderBuilder.append("(");
            for (OffloadDataEncoding placement : input) {
                if (((GPUParallelCallMapping) call).getOutputDataEncodings().stream().map(x -> x.getData()).allMatch(x -> x != placement.getData())) {
                    Data data = placement.getData();
                    kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                    kernelHeaderBuilder.append("*");
                    kernelHeaderBuilder.append(" ");
                    kernelHeaderBuilder.append(placement.getData().getIdentifier());
                    kernelHeaderBuilder.append(", ");

                    gpuBuilder.append(placement.getData().getIdentifier());
                    gpuBuilder.append(", ");
                }
            }

            for (OffloadDataEncoding placement : output) {
                Data data = placement.getData();
                kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                kernelHeaderBuilder.append("*");
                kernelHeaderBuilder.append(" ");
                kernelHeaderBuilder.append(placement.getData().getIdentifier());
                kernelHeaderBuilder.append(", ");

                gpuBuilder.append(placement.getData().getIdentifier());
                gpuBuilder.append(", ");
            }

            gpuBuilder.deleteCharAt(gpuBuilder.length() - 2);
            kernelHeaderBuilder.deleteCharAt(kernelHeaderBuilder.length() - 2);

            kernelHeaderBuilder.append(")");
            gpuBuilder.append(");\n");

            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n\n");


            gpuKernelHeaderBuilder.append(kernelHeaderBuilder);
            gpuKernelHeaderBuilder.append(";\n\n");

            gpuBuilder.append(kernelHeaderBuilder);

            gpuBuilder.append(" {\n");
            addIndent();
            gpuIndent();

            gpuBuilder.append("int tid = blockIdx.x * blockDim.x + threadIdx.x;\n");

            // Define a cutoff for increasing the workload for a subset of threads on the gpu in order to compute the exact pattern.
            int range = (int) (lengths.get(0) / (((GPUParallelCallMapping) call).getNumBlocks() * ((GPUParallelCallMapping) call).getThreadsPerBlock()));
            int cutoff = range * (((GPUParallelCallMapping) call).getNumBlocks() * ((GPUParallelCallMapping) call).getThreadsPerBlock());
            boolean useCutoff = cutoff < lengths.get(0);

            gpuIndent();
            gpuBuilder.append("int exec_range = ");
            gpuBuilder.append(range);
            gpuBuilder.append(";\n");

            if (useCutoff) {
                gpuIndent();
                gpuBuilder.append("if (tid < ");
                gpuBuilder.append(lengths.get(0) - cutoff);
                gpuBuilder.append(") {\n");
                addIndent();
                gpuIndent();
                gpuBuilder.append("exec_range++;\n");
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
            }

            gpuIndent();
            gpuBuilder.append("for ( size_t range_iterator = ");
            gpuBuilder.append(starts.get(0) - haloStart);
            gpuBuilder.append("; range_iterator < exec_range + ");
            gpuBuilder.append(starts.get(0) - haloStart);
            gpuBuilder.append("; range_iterator++) {\n");
            addIndent();
            gpuIndent();
            String indexName = "INDEX0_" + call.getCallExpression().getInlineEnding();
            gpuBuilder.append("int ");
            gpuBuilder.append(indexName);
            gpuBuilder.append(" = tid * exec_range + range_iterator");
            if (useCutoff) {
                gpuBuilder.append(" + ");
                gpuBuilder.append(lengths.get(0) - cutoff);
            }
            gpuBuilder.append(";\n");
            if (useCutoff) {
                gpuIndent();
                gpuBuilder.append("if (tid < ");
                gpuBuilder.append(lengths.get(0) - cutoff);
                gpuBuilder.append(") {\n");
                addIndent();
                gpuIndent();
                gpuBuilder.append(indexName);
                gpuBuilder.append(" -= ");
                gpuBuilder.append(lengths.get(0) - cutoff);
                gpuBuilder.append(";\n");
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
            }

            for (int i = 1; i < dimensions; i++) {
                String index = "INDEX" + i + "_" + call.getCallExpression().getInlineEnding();

                gpuIndent();
                gpuBuilder.append("for (size_t ");
                gpuBuilder.append(index);
                gpuBuilder.append(" = ");
                gpuBuilder.append(starts.get(i));
                gpuBuilder.append("; ");
                gpuBuilder.append(index);
                gpuBuilder.append(" < ");
                gpuBuilder.append(starts.get(i));
                gpuBuilder.append(" + ");
                gpuBuilder.append(lengths.get(i));
                gpuBuilder.append("; ++");
                gpuBuilder.append(index);
                gpuBuilder.append(") {\n");
                addIndent();
            }

            for (OffloadDataEncoding placement : input) {
                placement.updateDataIdentifier();
            }

            // create gpu context
            onGPU = true;

            StringBuilder cpuBuilder = builder;
            builder = gpuBuilder;

            addVariables(function);

            for (MappingNode child : function.getChildren()) {
                child.accept(getRealThis());
            }

            closeScope(function);
            gpuBuilder = builder;
            builder = cpuBuilder;
            onGPU = false;

            for (int i = 1; i < dimensions; i++) {
                removeIndent();
                gpuIndent();
                gpuBuilder.append("}\n");
            }

            // end gpu context
            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n");
            for (OffloadDataEncoding placement: input ) {
                placement.resetDataIdentifier();
            }
            removeIndent();
            gpuIndent();
            gpuBuilder.append("}\n\n");

            numIndents = currentIndent;
        } else {

            int core_offset = 0;
            if (call.getExecutor().isFirstCG()) {
                lengths.set(0, (lengths.get(0)*call.getExecutor().getCores())/(call.getExecutor().getCores() - 1));
                core_offset = 1;
            }
            for (int j = core_offset; j < call.getExecutor().getCores(); j++) {
                String lambda = "f_" + j + "_" + call.getCallExpression().getInlineEnding();
                if (doLambda) {
                    createLambdaHeader(lambda, call);
                }

                if (j == call.getExecutor().getCores() - 1) {
                    lengths.set(0, end - starts.get(0));
                }
                for (int i = 0; i < dimensions; i++) {
                    String index = "INDEX" + i + "_" + call.getCallExpression().getInlineEnding();

                    indent();
                    builder.append("for (size_t ");
                    builder.append(index);
                    builder.append(" = ");
                    builder.append(starts.get(i));
                    builder.append("; ");
                    builder.append(index);
                    builder.append(" < ");
                    builder.append(starts.get(i));
                    builder.append(" + ");
                    builder.append(lengths.get(i));
                    builder.append("; ++");
                    builder.append(index);
                    builder.append(") {\n");
                    addIndent();
                }

                starts.set(0, starts.get(0) + lengths.get(0));
                addVariables(function);

                for (MappingNode child : function.getChildren()) {
                    child.accept(getRealThis());
                }

                closeScope(function);

                for (int i = 0; i < dimensions; i++) {
                    removeIndent();
                    indent();
                    builder.append("}\n");
                }

                if (doLambda) {
                    createLambdaEnd(lambda, call, j);
                }
            }
        }



        if (needsMPI) {
            removeIndent();
            indent();
            builder.append("}\n");
        }
        clearParallelVariableInliningTable(function);
    }

    /**
     * Generates a parallel version of the dynamic programming pattern.
     * @param call
     * @param function
     */
    private void generateDPParallel(ParallelCallMapping call, DynamicProgrammingMapping function, boolean doLambda) {
        if (!call.getGroup().isLastCall()) {
            return;
        }
        fillParallelVariableInliningTable(call,function);

        ArrayList<Long> lengths = new ArrayList<>(call.getNumIterations());
        ArrayList<Long> starts = new ArrayList<>(call.getStartIndex());
        long end = starts.get(1) + lengths.get(1);

        lengths.set(1,(long) lengths.get(1)/call.getExecutor().getCores());
        String indexOuter = "INDEX" + 0 + "_" + call.getCallExpression().getInlineEnding();
        // outer loop
        indent();
        builder.append("for (size_t ");
        builder.append(indexOuter);
        builder.append(" = ");
        builder.append(starts.get(0));
        builder.append("; ");
        builder.append(indexOuter);
        builder.append(" < ");
        builder.append(starts.get(0));
        builder.append(" + ");
        builder.append(lengths.get(0));
        builder.append("; ++");
        builder.append(indexOuter);
        builder.append(") {\n");
        addIndent();

        if (call.getGroup() instanceof CallGroup) {
            for (ParallelCallMapping currentCall : ((CallGroup) call.getGroup()).getGroup()) {
                String currentInlineEnding = RandomStringGenerator.getAlphaNumericString();
                starts = new ArrayList<>(currentCall.getStartIndex());
                if (currentCall instanceof GPUParallelCallMapping) {
                    String lambda = "f_gpu_" + currentInlineEnding;
                    String wrapper_name = "cuda_wrapper_" + call.getFunctionIdentifier() + "_" + currentInlineEnding;
                    lengths.set(1, currentCall.getNumIterations().get(1));
                    StringBuilder wrapper_header_builder = new StringBuilder();
                    wrapper_header_builder.append("void ");
                    wrapper_header_builder.append(wrapper_name);
                    wrapper_header_builder.append("(");

                    ArrayList<OffloadDataEncoding> input = ((GPUParallelCallMapping) currentCall).getInputDataEncodings();
                    ArrayList<OffloadDataEncoding> output = ((GPUParallelCallMapping) currentCall).getOutputDataEncodings();

                    if (needsMPI) {
                        indent();
                        builder.append("if (rank == ");
                        builder.append(currentCall.getExecutor().getParent().getParent().getRank());
                        builder.append(") {\n");
                        addIndent();
                    }
                    createLambdaHeaderDP(lambda, currentCall);

                    indent();
                    builder.append(wrapper_name);
                    builder.append("(");

                    // get parameters
                    for (OffloadDataEncoding value :  input) {
                        Data parameter = value.getData();
                        wrapper_header_builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                        wrapper_header_builder.append("* ");
                        wrapper_header_builder.append(parameter.getIdentifier());
                        wrapper_header_builder.append(", ");
                        if (!(parameter instanceof ArrayData)) {
                            builder.append("&");
                        }
                        builder.append(value.getIdentifier());
                        builder.append(", ");
                    }
                    for (OffloadDataEncoding value :  output) {
                        Data parameter = value.getData();
                        wrapper_header_builder.append(CppTypesPrinter.doPrintType(parameter.getTypeName()));
                        wrapper_header_builder.append("* ");
                        wrapper_header_builder.append(parameter.getIdentifier());
                        wrapper_header_builder.append(", ");
                        if (!(parameter instanceof ArrayData)) {
                            builder.append("&");
                        }
                        builder.append(value.getIdentifier());
                        builder.append(", ");
                    }
                    wrapper_header_builder.append("int INDEX0_");
                    wrapper_header_builder.append(call.getCallExpression().getInlineEnding());
                    wrapper_header_builder.append(")");
                    builder.append("INDEX0_");
                    builder.append(call.getCallExpression().getInlineEnding());
                    builder.append(");\n");

                    createGPULambdaEnd(lambda, currentCall.getExecutor().getParent().getGPUrank());

                    if (needsMPI) {
                        removeIndent();
                        indent();
                        builder.append("}\n");
                    }

                    String wrapper_header = wrapper_header_builder.toString();

                    gpuHeaderBuilder.append(wrapper_header);
                    gpuHeaderBuilder.append(";\n\n");

                    int currentIndent = numIndents;
                    numIndents = 0;
                    gpuBuilder.append(wrapper_header);
                    gpuBuilder.append(" {\n");
                    addIndent();

                    gpuBuilder.append("\n");

                    // compute the start of halo data to ensure the correct dependencies between in-/output data elements.
                    long haloStart = Long.MAX_VALUE;
                    for (OffloadDataEncoding placement : input) {
                        if (placement.getStart() < haloStart) {
                            haloStart = placement.getStart();
                        }
                    }
                    for (OffloadDataEncoding placement : output) {
                        if (placement.getStart() < haloStart) {
                            haloStart = placement.getStart();
                        }
                    }

                    String kernelName = "kernel_" + wrapper_name;
                    StringBuilder kernelHeaderBuilder = new StringBuilder("__global__ \nvoid " + kernelName);

                    gpuIndent();
                    gpuBuilder.append(kernelName);
                    gpuBuilder.append("<<<");
                    gpuBuilder.append(((GPUParallelCallMapping) currentCall).getNumBlocks());
                    gpuBuilder.append(", ");
                    gpuBuilder.append(((GPUParallelCallMapping) currentCall).getThreadsPerBlock());
                    gpuBuilder.append(">>> (");
                    kernelHeaderBuilder.append("(");
                    for (OffloadDataEncoding placement : input) {
                        Data data = placement.getData();
                        kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                        kernelHeaderBuilder.append("* ");
                        kernelHeaderBuilder.append(data.getIdentifier());
                        kernelHeaderBuilder.append(", ");

                        gpuBuilder.append(data.getIdentifier());
                        gpuBuilder.append(", ");
                    }

                    for (OffloadDataEncoding placement : output) {
                        Data data = placement.getData();
                        kernelHeaderBuilder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                        kernelHeaderBuilder.append("* ");
                        kernelHeaderBuilder.append(data.getIdentifier());
                        kernelHeaderBuilder.append(", ");

                        gpuBuilder.append(data.getIdentifier());
                        gpuBuilder.append(", ");
                    }

                    kernelHeaderBuilder.append("int INDEX0_");
                    kernelHeaderBuilder.append(call.getCallExpression().getInlineEnding());
                    gpuBuilder.append("INDEX0_");
                    gpuBuilder.append(call.getCallExpression().getInlineEnding());

                    kernelHeaderBuilder.append(")");
                    gpuBuilder.append(");\n");

                    removeIndent();
                    gpuIndent();
                    gpuBuilder.append("}\n\n");


                    gpuKernelHeaderBuilder.append(kernelHeaderBuilder);
                    gpuKernelHeaderBuilder.append(";\n\n");

                    gpuBuilder.append(kernelHeaderBuilder);

                    gpuBuilder.append(" {\n");
                    addIndent();
                    gpuIndent();

                    gpuBuilder.append("int tid = blockIdx.x * blockDim.x + threadIdx.x;\n");

                    // Define a cutoff for increasing the workload for a subset of threads on the gpu in order to compute the exact pattern.
                    int range = (int) (lengths.get(1) / (((GPUParallelCallMapping) currentCall).getNumBlocks() * ((GPUParallelCallMapping) currentCall).getThreadsPerBlock()));
                    int cutoff = range * (((GPUParallelCallMapping) currentCall).getNumBlocks() * ((GPUParallelCallMapping) currentCall).getThreadsPerBlock());
                    boolean useCutoff = cutoff < lengths.get(1);

                    gpuIndent();
                    gpuBuilder.append("int exec_range = ");
                    gpuBuilder.append(range);
                    gpuBuilder.append(";\n");

                    if (useCutoff) {
                        gpuIndent();
                        gpuBuilder.append("if (tid < ");
                        gpuBuilder.append(lengths.get(1) - cutoff);
                        gpuBuilder.append(") {\n");
                        addIndent();
                        gpuIndent();
                        gpuBuilder.append("exec_range++;\n");
                        removeIndent();
                        gpuIndent();
                        gpuBuilder.append("}\n");
                    }

                    gpuIndent();
                    gpuBuilder.append("for ( int range_iterator = ");
                    gpuBuilder.append(starts.get(1) - haloStart);
                    gpuBuilder.append("; range_iterator < exec_range + ");
                    gpuBuilder.append(starts.get(1) - haloStart);
                    gpuBuilder.append("; range_iterator++) {\n");
                    addIndent();
                    gpuIndent();
                    String indexName = "INDEX1_" + call.getCallExpression().getInlineEnding();
                    gpuBuilder.append("int ");
                    gpuBuilder.append(indexName);
                    gpuBuilder.append(" = tid * exec_range + range_iterator");
                    if (useCutoff) {
                        gpuBuilder.append(" + ");
                        gpuBuilder.append(lengths.get(1) - cutoff);
                    }
                    gpuBuilder.append(";\n");
                    if (useCutoff) {
                        gpuIndent();
                        gpuBuilder.append("if (tid < ");
                        gpuBuilder.append(lengths.get(1) - cutoff);
                        gpuBuilder.append(") {\n");
                        addIndent();
                        gpuIndent();
                        gpuBuilder.append(indexName);
                        gpuBuilder.append(" -= ");
                        gpuBuilder.append(lengths.get(1) - cutoff);
                        gpuBuilder.append(";\n");
                        removeIndent();
                        gpuIndent();
                        gpuBuilder.append("}\n");
                    }


                    for (OffloadDataEncoding placement : input) {
                        placement.updateDataIdentifier();
                    }

                    // create gpu context
                    onGPU = true;

                    StringBuilder cpuBuilder = builder;
                    builder = gpuBuilder;

                    addVariables(function);

                    for (MappingNode child : function.getChildren()) {
                        child.accept(getRealThis());
                    }

                    closeScope(function);
                    gpuBuilder = builder;
                    builder = cpuBuilder;
                    onGPU = false;
                    // end gpu context
                    removeIndent();
                    gpuIndent();
                    gpuBuilder.append("}\n");
                    for (OffloadDataEncoding placement: input ) {
                        placement.resetDataIdentifier();
                    }
                    removeIndent();
                    gpuIndent();
                    gpuBuilder.append("}\n\n");

                    numIndents = currentIndent;
                } else {

                    if (needsMPI) {
                        indent();
                        builder.append("if (rank == ");
                        builder.append(currentCall.getExecutor().getParent().getParent().getRank());
                        builder.append(") {\n");
                        addIndent();
                    }
                    starts = new ArrayList<>(currentCall.getStartIndex());
                    lengths = new ArrayList<>(currentCall.getNumIterations());
                    end = starts.get(1) + lengths.get(1);
                    lengths.set(1,(long) lengths.get(1)/call.getExecutor().getCores());

                    int core_offset = 0;
                    if (currentCall.getExecutor().isFirstCG()) {
                        lengths.set(1, (lengths.get(1)*call.getExecutor().getCores())/(call.getExecutor().getCores() - 1));
                        core_offset = 1;
                    }
                    for (int j = core_offset; j < currentCall.getExecutor().getCores(); j++) {
                        String lambda = "f_" + j + "_" + currentCall.getCallExpression().getInlineEnding();
                        if (doLambda) {
                            createLambdaHeader(lambda, currentCall);
                        }
                        if (j == currentCall.getExecutor().getCores() - 1) {
                            lengths.set(1, end - starts.get(1));
                        }
                        String index = "INDEX" + 1 + "_" + currentCall.getCallExpression().getInlineEnding();

                        indent();
                        builder.append("for (size_t ");
                        builder.append(index);
                        builder.append(" = ");
                        builder.append(starts.get(1));
                        builder.append("; ");
                        builder.append(index);
                        builder.append(" < ");
                        builder.append(starts.get(1));
                        builder.append(" + ");
                        builder.append(lengths.get(1));
                        builder.append("; ++");
                        builder.append(index);
                        builder.append(") {\n");
                        addIndent();

                        starts.set(1, starts.get(1) + lengths.get(1));
                        addVariables(function);

                        for (MappingNode child : function.getChildren()) {
                            child.accept(getRealThis());
                        }

                        closeScope(function);

                        removeIndent();
                        indent();
                        builder.append("}\n");


                        if (doLambda) {
                            createLambdaEnd(lambda, currentCall, j);
                        }


                    }

                    if (needsMPI) {
                        removeIndent();
                        indent();
                        builder.append("}\n");
                    }
                }
            }
        }

        for (ParallelCallMapping currentCall : ((CallGroup) call.getGroup()).getGroup()) {
            for (GPUDataMovementMapping movement : currentCall.getDpPreSwapTransfers()) {
                movement.accept(getRealThis());
            }
        }

        if (call.getDynamicProgrammingBarrier().isPresent()) {
            call.getDynamicProgrammingBarrier().get().accept(getRealThis());
        }

        for (AbstractDataMovementMapping movement: call.getDynamicProgrammingdataTransfers() ) {
            movement.accept(getRealThis());
        }

        indent();
        builder.append("Set_Partial_Array(");
        builder.append(call.getArgumentExpressions().get(0).getOperands().get(0).getIdentifier());
        builder.append(", ");
        builder.append(call.getOutputElements().get(0).getIdentifier());
        builder.append(", ");
        if (call.getOutputElements().get(0) instanceof ArrayData) {
            builder.append(((ArrayData) call.getOutputElements().get(0)).getShape().get(0));
        } else {
            builder.append("0");
        }
        builder.append(");\n");

        for (ParallelCallMapping currentCall : ((CallGroup) call.getGroup()).getGroup()) {
            for (GPUDataMovementMapping movement : currentCall.getDpPostSwapTransfers()) {
                movement.accept(getRealThis());
            }
        }
        //close last loop
        removeIndent();
        indent();
        builder.append("}\n");

        clearParallelVariableInliningTable(function);
    }

    /**
     * Generates a parallel version of the recursion pattern.
     * @param call
     * @param function
     */
    private void generateRecursionParallel(ParallelCallMapping call, RecursionMapping function) {
        indent();
        builder.append(CppExpressionPrinter.doPrintExpression(call.getDefinition().getExpression(), onGPU, aktiveFunctions, activePatterns));
        builder.append(";\n");
        addSimpleFunction(call, function);
        addSimpleFunctionGPU(call, function);
    }


    /*****************************************************************
     *
     *              Support Functions
     *
     ******************************************************************/

    public void addSimpleFunction(CallMapping call, FunctionMapping function) {
        if (!simpleFunctions.containsKey(call.getFunctionIdentifier())) {
            StringBuilder simpleBuilder = new StringBuilder();

            int oldIndent = numIndents;

            StringBuilder oldBuilder = builder;

            simpleBuilder.append(CppPlainNodePrinter.printFunctionType(function));
            simpleBuilder.append(" ");
            simpleBuilder.append(call.getFunctionIdentifier());
            simpleBuilder.append(CppPlainNodePrinter.printArguments(function));
            simpleBuilder.append(" {\n");

            numIndents = 1;
            builder = simpleBuilder;

            simpleFunctions.put(function.getIdentifier(), simpleBuilder);

            addVariables(function);
            if (function instanceof RecursionMapping) {
                indent();
                builder.append(CppTypesPrinter.doPrintType(((RecursionMapping) function).getReturnElement().getTypeName()));
                builder.append(" ");
                builder.append(((RecursionMapping) function).getReturnElement().getIdentifier());
                builder.append(";\n");
            }

            for (MappingNode child: function.getChildren()) {
                child.accept(getRealThis());
            }

            builder = oldBuilder;
            numIndents = oldIndent;
            simpleBuilder.append("}\n");
            simpleFunctions.put(function.getIdentifier(), simpleBuilder);
        }
    }

    public void addSimpleFunctionGPU(CallMapping call, FunctionMapping function) {
        if (!simpleFunctionsGPU.containsKey(call.getFunctionIdentifier())) {
            StringBuilder simpleBuilder = new StringBuilder();

            int oldIndent = numIndents;

            StringBuilder oldBuilder = builder;

            simpleBuilder.append(CppPlainNodePrinter.printFunctionType(function));
            simpleBuilder.append(" ");
            simpleBuilder.append(call.getFunctionIdentifier());
            simpleBuilder.append("");
            simpleBuilder.append(CppPlainNodePrinter.printArguments(function));
            simpleBuilder.append(" {\n");

            numIndents = 1;
            builder = simpleBuilder;

            simpleFunctionsGPU.put(function.getIdentifier(), simpleBuilder);

            addVariables(function);
            if (function instanceof RecursionMapping) {
                indent();
                builder.append(CppTypesPrinter.doPrintType(((RecursionMapping) function).getReturnElement().getTypeName()));
                builder.append(" ");
                builder.append(((RecursionMapping) function).getReturnElement().getIdentifier());
                builder.append(";\n");
            }
            for (MappingNode child: function.getChildren()) {
                child.accept(getRealThis());
            }
            builder = oldBuilder;
            numIndents = oldIndent;
            simpleBuilder.append("}\n");
            simpleFunctionsGPU.put(function.getIdentifier(), simpleBuilder);
        }
    }


    /**
     * Generates the internal variables for the scope spanned by node.
     * @param node
     */
    private void addVariables(MappingNode node) {
        if (node instanceof FunctionMapping) {
            for (Data data : node.getVariableTable().values()) {
                if ((!AMT.getGlobalVariableTable().values().contains(data) && !data.isParameter() && !data.isReturnData()) && !(data instanceof FunctionInlineData) && !(data instanceof TempData)) {
                    if (data instanceof ArrayData) {
                        if (((ArrayData) data).isOnStack()) {
                            continue;
                        }
                    }
                    data.setInlineIdentifier(RandomStringGenerator.getAlphaNumericString());
                    indent();
                    builder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                    if (data instanceof ArrayData) {
                        builder.append("*");
                    }
                    builder.append(" ");
                    builder.append(data.getIdentifier());
                    builder.append(";\n");
                }
            }
        } else if (node instanceof CallMapping) {
            FunctionMapping function = AbstractMappingTree.getFunctionTable().get(((CallMapping) node).getFunctionIdentifier());

            for (Data data : function.getVariableTable().values()) {
                if ((!node.getParent().getVariableTable().values().contains(data) && !data.isParameter() && !data.isReturnData()) && !(data instanceof FunctionInlineData)) {
                    if (data instanceof ArrayData) {
                        if (((ArrayData) data).isOnStack()) {
                            continue;
                        }
                    }
                    indent();
                    builder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                    if (data instanceof ArrayData) {
                        builder.append("*");
                    }
                    builder.append(" ");
                    builder.append(data.getIdentifier());
                    boolean doInline = calculateInlining((CallMapping) node);
                    if (doInline || onGPU) {
                        builder.append("_");
                        builder.append(((CallMapping) node).getCallExpression().getInlineEnding());
                    }

                    builder.append(";\n");
                }
            }
        } else {
            for (Data data : node.getVariableTable().values()) {
                if ((!node.getParent().getVariableTable().containsValue(data) && !data.isParameter() && !data.isReturnData()) && !(data instanceof FunctionInlineData)) {
                    if (data instanceof ArrayData) {
                        if (((ArrayData) data).isOnStack()) {
                            continue;
                        }
                    }
                    indent();
                    builder.append(CppTypesPrinter.doPrintType(data.getTypeName()));
                    if (data instanceof ArrayData) {
                        builder.append("*");
                    }
                    builder.append(" ");
                    builder.append(data.getIdentifier());
                    if (!aktiveFunctions.isEmpty()) {
                        if (aktiveFunctions.get(aktiveFunctions.size() - 1).getCallExpression().getInlineEnding() != null) {
                            builder.append("_");
                            builder.append(aktiveFunctions.get(aktiveFunctions.size() - 1).getCallExpression().getInlineEnding());
                        }
                    }
                    builder.append(";\n");
                }
            }
        }
    }

    /**
     * Generates the code for de-allocating all memory in the heap which was allocated in the scope of this node.
     * If a function returns an array it is not de-allocated.
     * @param node
     */
    private void closeScope(MappingNode node) {
        if (node instanceof FunctionMapping) {
            for (Data data : node.getVariableTable().values()) {
                if ((!AMT.getGlobalVariableTable().values().contains(data) && !data.isParameter() && !data.isReturnData()) && !(data instanceof FunctionInlineData)) {
                    if (data instanceof ArrayData) {
                        if (((ArrayData) data).isOnStack()) {
                            continue;
                        }
                        ArrayList<Data> globalScope = new ArrayList<>(AMT.getGlobalVariableTable().values());
                        if (!((ArrayData) data).isOnStack() && !globalScope.contains(data) && !data.isParameter()) {
                            if (!data.isClosed()) {
                                indent();
                                builder.append("std::free(");
                                builder.append(data.getIdentifier());
                                builder.append(");\n");
                                data.setClosed();
                            }
                        }
                    }
                    data.resetInlineIdentifier();
                }
            }
        } else if (node instanceof ReturnMapping) {
            if (node.getParent() instanceof MainMapping) {
                FunctionMapping functionNode = (FunctionMapping) node.getParent();
                ArrayList<Data> globalScope = new ArrayList<>(AMT.getGlobalVariableTable().values());
                Data returnData;
                if (node.getChildren().size() == 1 && ((ReturnMapping) node).getResult().isPresent()) {
                    returnData = ((OperationExpression) ((ReturnMapping) node).getResult().get().getExpression()).getOperands().get(0);
                } else {
                    returnData = null;
                }
                for (Data data : functionNode.getVariableTable().values()) {
                    if (data instanceof ArrayData) {
                        if (!((ArrayData) data).isOnStack() && !globalScope.contains(data) && data != returnData && !data.isParameter() && !data.isInlinedParameter() && !data.getIdentifier().startsWith("inlineReturn_")) {
                            if (!data.isClosed()) {
                                indent();
                                builder.append("std::free(");
                                builder.append(data.getIdentifier());
                                builder.append(");\n");
                                data.setClosed();
                            }
                        }
                    }
                }
            } else {
                CallMapping callNode = aktiveFunctions.get(aktiveFunctions.size() - 1);
                closeInlineScope((ReturnMapping) node, callNode);
            }
        } else {
            MappingNode parentNode = node.getParent();
            ArrayList<Data> parentScope = new ArrayList<>(parentNode.getVariableTable().values());
            for (Data data: node.getVariableTable().values()) {
                if (data instanceof ArrayData) {
                    if (!((ArrayData) data).isOnStack() && !parentScope.contains(data) && !data.isParameter()) {
                        if (!data.isClosed()) {
                            indent();
                            builder.append("std::free(");
                            builder.append(data.getIdentifier());
                            if (!aktiveFunctions.isEmpty()) {
                                if (aktiveFunctions.get(aktiveFunctions.size() - 1).getCallExpression().getInlineEnding() != null) {
                                    builder.append("_");
                                    builder.append(aktiveFunctions.get(aktiveFunctions.size() - 1).getCallExpression().getInlineEnding());
                                }
                            }
                            builder.append(");\n");
                            data.setClosed();
                        }
                    }
                }
            }
        }
    }

    /**
     * Generates the code for de-allocating all memory in the heap which was allocated in the scope of the current function.
     * If a function returns an array it is not de-allocated.
     * @param node
     */
    private void closeInlineScope(ReturnMapping node, CallMapping callNode) {
        Data returnData;
        if (node.getResult().isPresent()) {
            returnData = ((OperationExpression) node.getResult().get().getExpression()).getOperands().get(0);
        } else {
            returnData = null;
        }
        FunctionMapping functionNode = AbstractMappingTree.getFunctionTable().get(callNode.getFunctionIdentifier());
        ArrayList<Data> globalScope = new ArrayList<>(AMT.getGlobalVariableTable().values());
        for (Data data : functionNode.getVariableTable().values()) {
            if (data instanceof ArrayData) {
                if (!((ArrayData) data).isOnStack() && !globalScope.contains(data) && data != returnData && !data.isParameter()) {
                    if (!data.isClosed()) {
                        indent();
                        builder.append("std::free(");
                        builder.append(data.getIdentifier());
                        boolean doInline = calculateInlining(callNode);
                        if (doInline || onGPU) {
                            builder.append("_");
                            builder.append(callNode.getCallExpression().getInlineEnding());
                        }
                        builder.append(");\n");
                        data.setClosed();
                    }
                }
            }
        }
    }

    /**
     * prints the function type of a given function
     * @param function
     * @return
     */
    public static String printFunctionType(FunctionMapping function) {
        String result = "";
        if (function instanceof SerialMapping) {
            result += CppTypesPrinter.doPrintType(((SerialMapping) function).getReturnType());
            if (((SerialMapping) function).isList()) {
                result += "*";
            }
        } else if (function instanceof ParallelMapping) {
            result += CppTypesPrinter.doPrintType(((ParallelMapping) function).getReturnElement().getTypeName());
            if (((ParallelMapping) function).getReturnElement() instanceof ArrayData) {
                result += "*";
            }
        }
        return result;
    }

    /**
     * prints the arguments of a given function
     * @param function
     * @return
     */
    public static String printArguments(FunctionMapping function) {
        StringBuilder builder = new StringBuilder();
        builder.append("(");
        ArrayList<Data> parameters = function.getArgumentValues();

        for (int i = 0; i < parameters.size(); i++ ) {
            builder.append(CppTypesPrinter.doPrintType(parameters.get(i).getTypeName()));
            if (parameters.get(i) instanceof ArrayData) {
                builder.append("*");
            }
            builder.append(" " + parameters.get(i).getIdentifier());
            if (i < parameters.size() - 1) {
                builder.append(", ");
            }
        }

        builder.append(")");
        return builder.toString();
    }

    /**
     * Generates the set of simple funktions.
     * @return
     */
    public static HashMap<String, StringBuilder> getSimpleFunctions() {
        return instance.simpleFunctions;
    }


    /**
     * Generates the set of simple funktions.
     * @return
     */
    public static HashMap<String, StringBuilder> getSimpleFunctionsCuda() {
        HashMap<String, StringBuilder> result = new HashMap<>();
        for (Map.Entry<String, StringBuilder> entry : instance.simpleFunctionsGPU.entrySet()) {
            if (AbstractMappingTree.getFunctionTable().get(entry.getKey()) instanceof RecursionMapping) {
                entry.getValue().insert(0, "__host__ __device__\n");
            }
            result.put(entry.getKey(), entry.getValue());
        }

        return result;
    }

    /**
     * Increases the number of indents by one.
     */
    private void addIndent() {
        numIndents++;
    }

    /**
     * Reduces the number of indents by one.
     */
    private void removeIndent() {
        numIndents--;
    }

    /**
     * Adds the current number of indents to the builder.
     */
    private void indent() {
        for (int i = 0; i < numIndents; i++) {
            builder.append("\t");
        }
    }

    /**
     * Adds the current number of indents to the builder.
     */
    private void gpuIndent() {
        for (int i = 0; i < numIndents; i++) {
            gpuBuilder.append("\t");
        }
    }

    /**
     * Returns the string generated by this class.
     * @return
     */
    @Override
    public String toString() {
        return builder.toString();
    }

    /**
     * Returns the gpu header generated by this class.
     * @return
     */
    private String toStringGPUHeader() {
        return gpuHeaderBuilder.toString();
    }

    /**
     * Returns the gpu header generated by this class.
     * @return
     */
    private String toStringGPUKernelHeader() {
        return gpuKernelHeaderBuilder.toString();
    }

    /**
     * Returns the gpu source generated by this class.
     * @return
     */
    private String toStringGPU() {
        return gpuBuilder.toString();
    }

    /**
     * Returns true, iff the callnode node takes an array as an argument.
     * @param node
     * @return
     */
    private boolean hasListArguments(CallMapping node) {
        FunctionMapping function = AbstractMappingTree.getFunctionTable().get(node.getFunctionIdentifier());
        return function.getArgumentValues().stream().anyMatch(c -> c instanceof ArrayData);
    }
}
