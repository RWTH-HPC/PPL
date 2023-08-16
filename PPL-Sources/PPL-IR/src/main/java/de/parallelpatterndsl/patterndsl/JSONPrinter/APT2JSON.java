package de.parallelpatterndsl.patterndsl.JSONPrinter;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.StructurePrinter.NodeAndEdgePrinter.CallTreePrinter;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;
import org.javatuples.Pair;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.stream.Collectors;

public class APT2JSON implements ExtendedShapeAPTVisitor {

    /**
     * The APT to be transformed.
     */
    private AbstractPatternTree APT;

    private StringBuilder builder;

    private String filename;

    private int dataSplitSize;

    private int patternSplitSize;

    private int numIndent;

    private HashSet<ParallelCallNode> encounteredFunctions;


    public APT2JSON(AbstractPatternTree APT, String filename, int dataSplitSize, int patternSplitSize) {
        this.APT = APT;
        this.builder = new StringBuilder();
        this.filename = filename;
        this.numIndent = 1;
        this.dataSplitSize = dataSplitSize;
        this.patternSplitSize = patternSplitSize;
        encounteredFunctions = new HashSet<>();

        generateJSON();
    }

    public String toString() {
        return builder.toString();
    }

    private void addIndent() {
        numIndent++;
    }

    private void removeIndent() {
        numIndent--;
    }

    private void indent() {
        for (int i = 0; i < numIndent; i++) {
            builder.append("  ");
        }
    }

    private void generateJSON() {

        // Name
        indent();
        builder.append("\"Name\" : \"");
        builder.append(filename);
        builder.append("\",\n");


        //Root
        indent();
        builder.append("\"Root\" : {\n");
        addIndent();
        indent();
        APT.getRoot().accept(this.getRealThis());
        removeIndent();
        indent();
        builder.append("},\n");

        // Global Variables
        indent();
        builder.append("\"Global Variables\" : [\n");
        addIndent();
        generateGlobalVars();
        removeIndent();
        indent();
        builder.append("],\n");

        // Functions
        indent();
        builder.append("\"Functions\" : [\n");
        addIndent();
        for (FunctionNode node: AbstractPatternTree.getFunctionTable().values()  ) {
            indent();
            builder.append("{\n");
            addIndent();
            node.accept(this.getRealThis());
            removeIndent();
            indent();
            builder.append("},\n");
        }
        if (!AbstractPatternTree.getFunctionTable().values().isEmpty()) {
            builder.deleteCharAt(builder.length() - 2);
        }
        removeIndent();
        indent();
        builder.append("]\n");

    }

    private void generateGlobalVars() {
        for (Data variable: APT.getGlobalVariableTable().values()) {
            if (!variable.getIdentifier().startsWith("inline")) {
                indent();
                builder.append("{\n");
                addIndent();
                generateVariable(variable);
                removeIndent();
                indent();
                builder.append("},\n");
            }
        }
        if (APT.getGlobalVariableTable().values().stream().anyMatch( x -> !x.getIdentifier().startsWith("inline"))) {
            builder.deleteCharAt(builder.length() - 2);
        }
        builder.append("\n");
    }

    private void generateVariable(Data variable) {

        //Name
        indent();
        builder.append("\"Name\" : \"");
        builder.append(variable.getIdentifier());
        builder.append("\",\n");
        //Data Type
        indent();
        builder.append("\"Data Type\" : \"");
        builder.append(PrimitiveDataTypes.toString(variable.getTypeName()));
        builder.append("\",\n");
        //Dimension
        indent();
        builder.append("\"Dimension\" : \"");
        if (variable instanceof PrimitiveData) {
            builder.append("0\"");
        } else if (variable instanceof ArrayData) {
            builder.append(((ArrayData) variable).getShape().size());
            builder.append("\"");
        }
        //Shape
        if (variable instanceof ArrayData && !variable.isParameter()) {
            builder.append(",\n");
            indent();
            builder.append("\"Shape\" : [");
            for (int size :((ArrayData) variable).getShape() ) {
                builder.append("\"");
                builder.append(size);
                builder.append("\",");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("]");
        }
        builder.append("\n");
    }

    private void generateParameters(FunctionNode node, boolean isInput) {
        ArrayList<Data> parameters;

        if (isInput) {
            parameters = node.getArgumentValues();
        } else if (node instanceof ParallelNode){
            parameters = new ArrayList<>();
            parameters.add(((ParallelNode) node).getReturnElement());
        } else {
            parameters = new ArrayList<>();
        }

        for (Data parameter: parameters ) {
            indent();
            builder.append("{\n");
            addIndent();
            generateVariable(parameter);
            removeIndent();
            indent();
            builder.append("},\n");
        }
        if (!parameters.isEmpty()) {
            builder.deleteCharAt(builder.length() - 2);
        }
    }

    private void generateScope(PatternNode node) {

        for (Data variable: node.getVariableTable().values() ) {
            if (!variable.isParameter() && !variable.getIdentifier().startsWith("inline")) {
                indent();
                builder.append("{\n");
                addIndent();
                generateVariable(variable);
                removeIndent();
                indent();
                builder.append("},");
            }
        }
        if (node.getVariableTable().values().stream().anyMatch(x -> !x.isParameter() && !x.getIdentifier().startsWith("inline"))) {
            builder.deleteCharAt(builder.length() - 1);
        }

    }

    private void generateDataAccess(DataAccess access) {

        //Variable
        indent();
        builder.append("\"Variable\" : \"");
        builder.append(access.getData().getIdentifier());
        builder.append("\",\n");

        //Write
        indent();
        builder.append("\"Write\" : \"");
        builder.append(!access.isReadAccess());
        builder.append("\",\n");

        //Type
        indent();
        builder.append("\"Type\" : \"");
        if (access instanceof MapDataAccess) {
            builder.append("Map\",");
            builder.append("\n");

            // Scaling
            indent();
            builder.append("\"Scaling\" : [\"");
            builder.append(((MapDataAccess) access).getScalingFactor());
            builder.append("\"],\n");

            //Shift
            indent();
            builder.append("\"Shift\" : [\"");
            builder.append(((MapDataAccess) access).getShiftOffset());
            builder.append("\"],\n");

            //Iterator
            indent();
            builder.append("\"Iterator\" : [\"INDEX\"]\n");

        } else if (access instanceof ReduceDataAccess) {
            builder.append("Reduce\",");
            builder.append("\n");

            // Scaling
            indent();
            builder.append("\"Scaling\" : [\"1\"],\n");

            //Shift
            indent();
            builder.append("\"Shift\" : [\"0\"],\n");

            //Iterator
            indent();
            builder.append("\"Iterator\" : [\"INDEX\"]\n");
        } else if (access instanceof DynamicProgrammingDataAccess) {
            builder.append("Dynamic Programming\",");
            builder.append("\n");

            // Scaling
            indent();
            builder.append("\"Scaling\" : [\"");
            for (String scalingFactor: ((DynamicProgrammingDataAccess) access).getRuleBaseIndex() ) {
                builder.append("\"1\",");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("],\n");

            //Shift
            indent();
            builder.append("\"Shift\" : [");
            for (int shiftOffset: ((DynamicProgrammingDataAccess) access).getShiftOffsets() ) {
                builder.append("\"");
                builder.append(shiftOffset);
                builder.append("\",");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("],\n");

            //Iterator
            indent();
            builder.append("\"Iterator\" : [");
            for (String Index: ((DynamicProgrammingDataAccess) access).getRuleBaseIndex() ) {
                builder.append("\"");
                builder.append(Index);
                builder.append("\",");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("]\n");
        } else if (access instanceof StencilDataAccess) {
            builder.append("Stencil\",");
            builder.append("\n");

            // Scaling
            indent();
            builder.append("\"Scaling\" : [");
            for (int scalingFactor: ((StencilDataAccess) access).getScalingFactors() ) {
                builder.append("\"");
                builder.append(scalingFactor);
                builder.append("\",");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("],\n");

            //Shift
            indent();
            builder.append("\"Shift\" : [");
            for (int shiftOffset: ((StencilDataAccess) access).getShiftOffsets() ) {
                builder.append("\"");
                builder.append(shiftOffset);
                builder.append("\",");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("],\n");

            //Iterator
            indent();
            builder.append("\"Iterator\" : [");
            for (String Index: ((StencilDataAccess) access).getRuleBaseIndex() ) {
                builder.append("\"");
                builder.append(Index);
                builder.append("\",");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("]\n");
        } else {
            builder.append("Serial\"");
            builder.append("\n");
        }
    }

    private void generateFunctionNode(FunctionNode node, String type) {
        //Name
        indent();
        builder.append("\"Name\" : \"");
        builder.append(node.getIdentifier());
        builder.append("\",\n");

        //Pattern
        indent();
        builder.append("\"Pattern\" : \"");
        builder.append(type);
        builder.append("\",\n");

        //Input
        indent();
        builder.append("\"Input\" : [\n");
        addIndent();
        generateParameters(node, true);
        removeIndent();
        indent();
        builder.append("],\n");

        //Output
        indent();
        builder.append("\"Output\" : [\n");
        addIndent();
        generateParameters(node, false);
        removeIndent();
        indent();
        builder.append("],\n");

        //Cost
        indent();
        builder.append("\"Cost\" : \"");
        builder.append(node.getCost());
        builder.append("\",\n");

        //Load/Store
        indent();
        builder.append("\"Load/Store\" : \"");
        builder.append(node.getLoadStore());
        builder.append("\",\n");

        //Variables
        indent();
        builder.append("\"Variables\" : [\n");
        addIndent();
        generateScope(node);
        removeIndent();
        indent();
        builder.append("],\n");

        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\",\n");

        //Children
        handleChildren(0, node);
    }

    private void handleChildren(int offset, PatternNode node) {
        indent();
        builder.append("\"Children\" : [\n");
        addIndent();
        for (int i = offset; i < node.getChildren().size(); i++) {
            PatternNode child = node .getChildren().get(i);
            indent();
            builder.append("{\n");
            addIndent();
            child.accept(this.getRealThis());
            removeIndent();
            indent();
            builder.append("},\n");
        }
        builder.deleteCharAt(builder.length() - 2);
        removeIndent();
        indent();
        builder.append("]\n");
    }

    /**
     * Tests if the current pattern node has a data dependency to the current step set.
     * @param set
     * @param node
     * @return
     */
    private boolean testDependency(Set<PatternNode> set, PatternNode node) {
        for (PatternNode element: set ) {
            ArrayList<Data> RbW = dataOverlap(element.getInputElements(), node.getOutputElements());
            ArrayList<Data> WbR = dataOverlap(element.getOutputElements(), node.getInputElements());
            ArrayList<Data> WbW = dataOverlap(element.getOutputElements(), node.getOutputElements());
            // Read before Write
            if (!RbW.isEmpty()) {
                if (!(node instanceof ParallelCallNode && element instanceof ParallelCallNode)) {
                    return true;
                } else {
                    for (Data data : RbW ) {
                        if (dataWiseOverlap(data, (ParallelCallNode) element, true, (ParallelCallNode) node, false)) {
                            return true;
                        }
                    }
                }
            } // Write before Read
            if (!WbR.isEmpty()) {
                if (!(node instanceof ParallelCallNode && element instanceof ParallelCallNode)) {
                    return true;
                } else {
                    for (Data data : RbW ) {
                        if (dataWiseOverlap(data, (ParallelCallNode) element, false, (ParallelCallNode) node, true)) {
                            return true;
                        }
                    }
                }
            } // Write after Write
            if (!WbW.isEmpty()) {
                if (!(node instanceof ParallelCallNode && element instanceof ParallelCallNode)) {
                    return true;
                } else {
                    for (Data data : RbW ) {
                        if (dataWiseOverlap(data, (ParallelCallNode) element, false, (ParallelCallNode) node, false)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    /**
     * Returns true iff the parallel call nodes access overlapping dataSplits
     * @param dataElement
     * @param node1
     * @param isInput1
     * @param node2
     * @param isInput2
     * @return
     */
    private boolean dataWiseOverlap(Data dataElement, ParallelCallNode node1, boolean isInput1, ParallelCallNode node2, boolean isInput2) {
        ParallelNode function1 = (ParallelNode) AbstractPatternTree.getFunctionTable().get(node1.getFunctionIdentifier());
        ParallelNode function2 = (ParallelNode) AbstractPatternTree.getFunctionTable().get(node2.getFunctionIdentifier());

        // Compute ranges

        Pair<ArrayList<Long>, ArrayList<Long>> indexRange1 = computeMaximumIndexRange(node1, function1);
        Pair<ArrayList<Long>, ArrayList<Long>> indexRange2 = computeMaximumIndexRange(node2, function2);

        ArrayList<Long> starts1 = indexRange1.getValue0();
        ArrayList<Long> lengths1 = indexRange1.getValue1();
        ArrayList<Long> starts2 = indexRange2.getValue0();
        ArrayList<Long> lengths2 = indexRange2.getValue1();

        ArrayList<Data> parameters1 = new ArrayList<>();
        ArrayList<Data> parameters2 = new ArrayList<>();

        if (isInput1) {
            for (int index: getIndexInput(dataElement, node1) ) {
                parameters1.add(function1.getArgumentValues().get(index));
            }
        } else {
            //Replace when multi-output is supported
            /*for (int index: getIndexOutput(dataElement, node1) ) {
                parameters1.add(function1.getArgumentValues().get(index));
            }*/
            parameters1.add(function1.getReturnElement());
        }

        if (isInput2) {
            for (int index: getIndexInput(dataElement, node2) ) {
                parameters2.add(function2.getArgumentValues().get(index));
            }
        } else {
            //Replace when multi-output is supported
            /*for (int index: getIndexOutput(dataElement, node2) ) {
                parameters2.add(function2.getArgumentValues().get(index));
            }*/
            parameters2.add(function2.getReturnElement());
        }

        if (!testShapes(parameters1, parameters2)) {
            return true;
        }

        ArrayList<DataAccess> accesses1 = getAccesses(function1, parameters1, isInput1);
        ArrayList<DataAccess> accesses2 = getAccesses(function2, parameters2, isInput2);

        return allAccessOverlap(starts1, starts2, lengths1, lengths2, accesses1, accesses2);
    }

    /**
     * Returns the data accesses corresponding to a given set of function parameters.
     * @param function
     * @param parameters
     * @param isInput
     * @return
     */
    private ArrayList<DataAccess> getAccesses(FunctionNode function, ArrayList<Data> parameters, boolean isInput) {
        ArrayList<DataAccess> result = new ArrayList<>();
        if (isInput) {
            for (Data parameter: parameters) {
                result.addAll(function.getInputAccesses().stream().filter(x -> x.getData() == parameter).collect(Collectors.toList()));
            }
        } else {
            for (Data parameter: parameters) {
                result.addAll(function.getOutputAccesses().stream().filter(x -> x.getData() == parameter).collect(Collectors.toList()));
            }
        }
        return result;
    }

    /**
     * Tests if all parameters share the same dimensionality.
     * @param parameters1
     * @param parameters2
     * @return
     */
    private boolean testShapes(ArrayList<Data> parameters1, ArrayList<Data> parameters2) {
        if (parameters1.get(0) instanceof PrimitiveData) {
            for (Data element: parameters1 ) {
                if (!(element instanceof PrimitiveData)) {
                    return false;
                }
            }
            for (Data element: parameters2 ) {
                if (!(element instanceof PrimitiveData)) {
                    return false;
                }
            }
        } else if(parameters1.get(0) instanceof ArrayData) {
            int dimension = ((ArrayData) parameters1.get(0)).getShape().size();
            for (Data element: parameters1 ) {
                if (!(element instanceof ArrayData)) {
                    return false;
                } else if (((ArrayData) element).getShape().size() != dimension) {
                    return false;
                }
            }
            for (Data element: parameters2 ) {
                if (!(element instanceof ArrayData)) {
                    return false;
                } else if (((ArrayData) element).getShape().size() != dimension) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Returns for which output argument the data element is used.
     * @param dataElement
     * @param node
     * @return
     */
    private ArrayList<Integer> getIndexOutput(Data dataElement, ParallelCallNode node) {
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i < node.getOutputElements().size(); i++) {
            if (node.getOutputElements().get(i) == dataElement) {
                result.add(i);
            }
        }
        return result;
    }

    /**
     * Returns for which arguments the given data element is used.
     * @param dataElement
     * @param node
     * @return
     */
    private ArrayList<Integer> getIndexInput(Data dataElement, ParallelCallNode node) {
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i < node.getArgumentExpressions().size(); i++) {
            if (node.getArgumentExpressions().get(i).getOperands().contains(dataElement)) {
                result.add(i);
            }
        }
        return result;
    }

    /**
     * Computes the maximum index range for all calls off the given function. The result is defined as <starts, lengths>
     * @param node
     * @param function
     * @return
     */
    private Pair<ArrayList<Long>, ArrayList<Long>> computeMaximumIndexRange(ParallelCallNode node, FunctionNode function) {
        ArrayList<Long> starts = new ArrayList<>();
        ArrayList<Long> lengths = new ArrayList<>();

        // For additional argument offsets please refer to the Wiki Page on APTs to define the order and size
        if (function instanceof MapNode) {
            long start = Long.MAX_VALUE;
            long end = Long.MIN_VALUE;
            for (int i = 0; i < node.getAdditionalArguments().size() / node.getAdditionalArgumentCount(); i++) {
                start =  Long.min(start, ((MetaValue<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 1)).getValue());
                end = Long.max(end, ((MetaValue<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 1)).getValue() + ((MetaValue<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount())).getValue());
            }
            starts.add(start);
            lengths.add(end - start);

        } else if (function instanceof ReduceNode) {
            long start = Long.MAX_VALUE;
            long end = Long.MIN_VALUE;
            for (int i = 0; i < node.getAdditionalArguments().size() / node.getAdditionalArgumentCount(); i++) {
                start =  Long.min(start, ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount())).getValues().get(4));
                end = Long.max(end, ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount())).getValues().get(4) + ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount())).getValues().get(0));
            }
            starts.add(start);
            lengths.add(end - start);
        } else if (function instanceof DynamicProgrammingNode) {
            long start = Long.MAX_VALUE;
            long end = Long.MIN_VALUE;
            long startRecursion = Long.MAX_VALUE;
            long endRecursion = Long.MIN_VALUE;

            for (int i = 0; i < node.getAdditionalArguments().size() / node.getAdditionalArgumentCount(); i++) {
                start =  Long.min(start, ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 2)).getValues().get(1));
                startRecursion =  Long.min(startRecursion, ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 2)).getValues().get(0));
                end = Long.max(end, ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 2)).getValues().get(1) + ((MetaValue<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 1)).getValue());
                endRecursion = Long.max(endRecursion, ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 2)).getValues().get(0) + ((MetaValue<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount())).getValue());
            }

            starts.add(startRecursion);
            lengths.add(endRecursion - startRecursion);

            starts.add(start);
            lengths.add(end - start);
        } else if (function instanceof StencilNode) {
            ArrayList<Long> start = new ArrayList<>();
            ArrayList<Long> end = new ArrayList<>();
            for (int i = 0; i < ((StencilNode) function).getDimension(); i++) {
                start.add(Long.MAX_VALUE);
                end.add(Long.MIN_VALUE);
            }
            for (int i = 0; i < node.getAdditionalArguments().size() / node.getAdditionalArgumentCount(); i++) {
                for (int j = 0; j < ((StencilNode) function).getDimension(); j++) {
                    long newStart =  Long.min(start.get(j), ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 1)).getValues().get(j));
                    start.set(j,newStart);
                    long newEnd =  Long.min(end.get(j), ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount() + 1)).getValues().get(j) + ((MetaList<Long>)node.getAdditionalArguments().get(i * node.getAdditionalArgumentCount())).getValues().get(j)) ;
                    end.set(j,newEnd);
                }
            }

            starts.addAll(start);
            for (int i = 0; i < ((StencilNode) function).getDimension(); i++) {
                lengths.add(end.get(i) - start.get(i));
            }
        }

        return new Pair<>(starts, lengths);
    }

    /**
     * Checks if for one data element the two parallel patterns have overlapping data accesses.
     * @param start1
     * @param start2
     * @param length1
     * @param length2
     * @param accesses1
     * @param accesses2
     * @return
     */
    private boolean allAccessOverlap(ArrayList<Long> start1, ArrayList<Long> start2, ArrayList<Long> length1, ArrayList<Long> length2, ArrayList<DataAccess> accesses1, ArrayList<DataAccess> accesses2) {
        for (DataAccess access1: accesses1 ) {
            for (DataAccess access2: accesses2 ) {
                if (accessOverlap(start1,start2,length1,length2,access1,access2)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Checks for two data accesses within a parallel pattern if their access pattern and index range create overlapping data.
     * @param start1
     * @param start2
     * @param length1
     * @param length2
     * @param access1
     * @param access2
     * @return
     */
    private boolean accessOverlap(ArrayList<Long> start1, ArrayList<Long> start2, ArrayList<Long> length1, ArrayList<Long> length2, DataAccess access1, DataAccess access2) {

        // test data access types, parallel reduction must always be the last entry of a reduction
        if (!(access1 instanceof MapDataAccess || access1 instanceof StencilDataAccess || access1 instanceof DynamicProgrammingDataAccess || access2 instanceof MapDataAccess || access2 instanceof StencilDataAccess || access2 instanceof DynamicProgrammingDataAccess)) {
            return true;
        }

        Set<Long> set1;
        Set<Long> set2;

        boolean markOverlap = false;

        if (start1.size() >= 1 && start2.size() >= 1 && length1.size() >= 1 && length2.size() >= 1  ) {
            set1 = generateOverlapSet(start1,length1,access1,0);

            set2 = generateOverlapSet(start2,length2,access2,0);

            for (Long testing: set1 ) {
                if (set2.contains(testing)) {
                    markOverlap = true;
                    break;
                }
            }

            if (start1.size() >= 2 && start2.size() >= 2 && length1.size() >= 2 && length2.size() >= 2  && markOverlap) {
                markOverlap = false;
                set1 = generateOverlapSet(start1,length1,access1,1);

                set2 = generateOverlapSet(start2,length2,access2,1);

                for (Long testing: set1 ) {
                    if (set2.contains(testing)) {
                        markOverlap = true;
                        break;
                    }
                }

                if (start1.size() >= 3 && start2.size() >= 3 && length1.size() >= 3 && length2.size() >= 3  && markOverlap) {
                    markOverlap = false;
                    set1 = generateOverlapSet(start1, length1, access1, 2);

                    set2 = generateOverlapSet(start2, length2, access2, 2);

                    for (Long testing : set1) {
                        if (set2.contains(testing)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    /**
     * Generates the set of accessed data splits.
     * @param start
     * @param length
     * @param access
     * @param depth
     * @return
     */
    private Set<Long> generateOverlapSet(ArrayList<Long> start, ArrayList<Long> length, DataAccess access, int depth) {
        Set<Long> set = new HashSet<>();
        for (long i = 0; i < length.get(0); i++) {
            if (access instanceof MapDataAccess && depth == 0) {
                set.add((((MapDataAccess) access).getScalingFactor() * (start.get(depth) + length.get(depth)) + ((MapDataAccess) access).getShiftOffset()) % dataSplitSize);
            } else if (access instanceof DynamicProgrammingDataAccess && depth <= 1) {
                if (((DynamicProgrammingDataAccess) access).getRuleBaseIndex().get(depth).equals("INDEX0")) {
                    set.add((start.get(0) + length.get(0) + ((DynamicProgrammingDataAccess) access).getShiftOffsets().get(depth)) % dataSplitSize);
                } else if (((DynamicProgrammingDataAccess) access).getRuleBaseIndex().get(depth).equals("INDEX1")) {
                    set.add((start.get(1) + length.get(1) + ((DynamicProgrammingDataAccess) access).getShiftOffsets().get(depth)) % dataSplitSize);
                }
            } else if (access instanceof StencilDataAccess) {
                int rangeLevel = Integer.parseInt(((StencilDataAccess) access).getRuleBaseIndex().get(depth).substring(5));
                set.add((((StencilDataAccess) access).getScalingFactors().get(depth) * (start.get(rangeLevel) + length.get(rangeLevel)) + ((StencilDataAccess) access).getShiftOffsets().get(depth)) % dataSplitSize);
            }
        }
        return set;
    }

    private ArrayList<Data> dataOverlap(ArrayList<Data> input1, ArrayList<Data> input2) {
        ArrayList<Data> result = new ArrayList<>();
        for (Data iteration1 : input1 ) {
            for (Data iteration2 : input2 ) {
                if (iteration1 == iteration2) {
                    result.add(iteration1);
                }
            }
        }
        return result;
    }

    private void generateNodeBody(String pattern, PatternNode node) {
        //Pattern
        indent();
        builder.append("\"Pattern Node Type\" : \"");
        builder.append(pattern);
        builder.append("\",\n");

        //Input
        indent();
        builder.append("\"InputData\" : [\n");
        addIndent();
        for (Data data: node.getInputElements()) {
            indent();
            builder.append("{\n");
            addIndent();
            generateVariable(data);
            removeIndent();
            indent();
            builder.append("},\n");
        }
        if (!node.getInputElements().isEmpty()) {
            builder.deleteCharAt(builder.length() - 2);
        }
        removeIndent();
        indent();
        builder.append("],\n");

        //Output
        indent();
        builder.append("\"OutputData\" : [\n");
        addIndent();
        for (Data data: node.getOutputElements()) {
            indent();
            builder.append("{\n");
            addIndent();
            generateVariable(data);
            removeIndent();
            indent();
            builder.append("},\n");
        }
        if (!node.getOutputElements().isEmpty()) {
            builder.deleteCharAt(builder.length() - 2);
        }
        removeIndent();
        indent();
        builder.append("],\n");

        //Input Accesses
        indent();
        builder.append("\"InputAccesses\" : [\n");
        addIndent();
        for (DataAccess data: node.getInputAccesses()) {
            indent();
            builder.append("{\n");
            addIndent();
            generateDataAccess(data);
            removeIndent();
            indent();
            builder.append("},\n");
        }
        if (!node.getInputAccesses().isEmpty()) {
            builder.deleteCharAt(builder.length() - 2);
        }
        removeIndent();
        indent();
        builder.append("],\n");

        //Output Accesses
        indent();
        builder.append("\"OutputAccesses\" : [\n");
        addIndent();
        for (DataAccess data: node.getOutputAccesses()) {
            indent();
            builder.append("{\n");
            addIndent();
            generateDataAccess(data);
            removeIndent();
            indent();
            builder.append("},\n");
        }
        if (!node.getOutputAccesses().isEmpty()) {
            builder.deleteCharAt(builder.length() - 2);
        }
        removeIndent();
        indent();
        builder.append("],\n");

        //Cost
        indent();
        builder.append("\"Cost\" : \"");
        builder.append(node.getCost());
        builder.append("\",\n");

        //Load/Store
        indent();
        builder.append("\"Load/Store\" : \"");
        builder.append(node.getLoadStore());
        builder.append("\",\n");

        //ID
        indent();
        builder.append("\"ID\" : \"");
        String id = RandomStringGenerator.getAlphaNumericString();
        NodeIDMapping.addMapping(id, node);
        builder.append(id);
        builder.append("\",\n");
    }

    @Override
    public void traverse(SerialNode node) {
        generateFunctionNode(node, "Serial");
    }

    @Override
    public void traverse(MapNode node) {
        generateFunctionNode(node, "Map");
    }

    @Override
    public void traverse(ReduceNode node) {
        generateFunctionNode(node, "Reduction");
    }

    @Override
    public void traverse(RecursionNode node) {
        generateFunctionNode(node, "Recursion");
    }


    @Override
    public void traverse(StencilNode node) {
        generateFunctionNode(node, "Stencil");
    }

    @Override
    public void traverse(DynamicProgrammingNode node) {
        generateFunctionNode(node, "Dynamic Programming");
    }


    @Override
    public void traverse(BranchNode node) {

        generateNodeBody("BranchNode", node);

        //Variables
        indent();
        builder.append("\"Variables\" : [\n");
        addIndent();
        generateScope(node);
        removeIndent();
        builder.append("],\n");

        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\",\n");

        //Children
        indent();
        builder.append("\"Children\" : [\n");
        addIndent();
        for (PatternNode child: node.getChildren()) {
            indent();
            builder.append("{\n");
            addIndent();
            child.accept(this.getRealThis());
            removeIndent();
            indent();
            builder.append("},\n");
        }
        builder.deleteCharAt(builder.length() - 2);
        removeIndent();
        builder.append("]\n");
    }

    @Override
    public void traverse(BranchCaseNode node) {
        generateNodeBody("BranchCaseNode", node);

        //Variables
        indent();
        builder.append("\"Variables\" : [\n");
        addIndent();
        generateScope(node);
        removeIndent();
        builder.append("],\n");

        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\",\n");

        // Condition
        if (node.isHasCondition()) {
            indent();
            builder.append("\"Condition\" : {");
            node.getChildren().get(0).accept(this.getRealThis());
            builder.append("},\n");
        }

        //Children
        if (node.isHasCondition()) {
            handleChildren(1, node);
        } else {
            handleChildren(0, node);
        }
    }

    @Override
    public void traverse(SimpleExpressionBlockNode node) {
        generateNodeBody("SimpleExpressionBlockNode", node);
        builder.deleteCharAt(builder.length() - 1);
        builder.deleteCharAt(builder.length() - 1);
    }

    @Override
    public void traverse(ComplexExpressionNode node) {

        generateNodeBody("ComplexExpressionNode", node);
        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\",\n");

        //Calls
        indent();
        builder.append("\"Calls\" : [\n");
        addIndent();
        for (int i = 0; i < node.getChildren().size(); i++) {
            PatternNode child = node.getChildren().get(i);
            if (!PredefinedFunctions.contains(((CallNode) child).getFunctionIdentifier())) {
                indent();
                builder.append("{\n");
                addIndent();
                child.accept(this.getRealThis());
                removeIndent();
                indent();
                builder.append("}");
                if (i + 1 < node.getChildren().size()) {
                    builder.append(",");
                }
                builder.append("\n");
            }
        }
        removeIndent();
        indent();
        builder.append("]\n");

    }


    @Override
    public void traverse(CallNode node) {
        generateNodeBody("CallNode", node);

        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\"\n");
    }

    @Override
    public void traverse(ForLoopNode node) {
        generateNodeBody("ForLoopNode", node);

        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\",\n");

        //Variables
        indent();
        builder.append("\"Variables\" : [\n");
        addIndent();
        generateScope(node);
        removeIndent();
        builder.append("],\n");

        //Loop Variable
        indent();
        builder.append("\"Loop Variable\" : {");
        generateVariable(node.getLoopControlVariable());
        builder.append("},\n");

        //Init
        indent();
        builder.append("\"Init\" : {");
        node.getChildren().get(0).accept(this.getRealThis());
        builder.append("},\n");

        //Condition
        indent();
        builder.append("\"Condition\" : {");
        node.getChildren().get(1).accept(this.getRealThis());
        builder.append("},\n");

        //Update
        indent();
        builder.append("\"Update\" : {");
        node.getChildren().get(2).accept(this.getRealThis());
        builder.append("},\n");

        //Children
        handleChildren(3, node);
    }

    @Override
    public void traverse(ForEachLoopNode node) {
        generateNodeBody("ForEachLoopNode", node);

        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\",\n");

        //Variables
        indent();
        builder.append("\"Variables\" : [\n");
        addIndent();
        generateScope(node);
        removeIndent();
        builder.append("],\n");

        //Loop Variable
        indent();
        builder.append("\"Loop Variable\" : {");
        generateVariable(node.getLoopControlVariable());
        builder.append("},\n");

        //Array
        indent();
        builder.append("\"Array\" : {");
        node.getChildren().get(0).accept(this.getRealThis());
        builder.append("},\n");

        //Children
        handleChildren(1, node);
    }

    @Override
    public void traverse(WhileLoopNode node) {
        generateNodeBody("WhileLoopNode", node);

        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\",\n");

        //Variables
        indent();
        builder.append("\"Variables\" : [\n");
        addIndent();
        generateScope(node);
        removeIndent();
        builder.append("],\n");

        //Condition
        indent();
        builder.append("\"Condition\" : {");
        node.getChildren().get(0).accept(this.getRealThis());
        builder.append("},\n");

        //Children
        handleChildren(1, node);
    }

    public void traverse(ReturnNode node) {
        generateNodeBody("ReturnNode", node);

        //Parallel Descendants
        indent();
        builder.append("\"Parallel Descendants\" : \"");
        builder.append(node.isHasParallelDescendants());
        builder.append("\",\n");

        //Condition
        indent();
        builder.append("\"Result\" : {\n");
        addIndent();
        node.getChildren().get(0).accept(this.getRealThis());
        removeIndent();
        indent();
        builder.append("}\n");

    }

    @Override
    public void traverse(ParallelCallNode node) {
        indent();
        builder.append("\"Name\" : \"");
        builder.append(node.getFunctionIdentifier());
        builder.append("\",\n");

        generateNodeBody("ParallelCallNode", node);

        //Starts
        indent();
        builder.append("\"Starts\" : [\n");
        addIndent();
        for (int i = 0; i < node.getAdditionalArguments().size() / node.getAdditionalArgumentCount(); i++) {
            FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
            indent();
            if (function instanceof MapNode) {
                builder.append("\"");
                builder.append(((MetaValue<Integer>) node.getAdditionalArguments().get(1)).getValue());
                builder.append("\"\n");
            } else if (function instanceof ReduceNode) {
                builder.append("\"");
                builder.append(((MetaList<Integer>) node.getAdditionalArguments().get(0)).getValues().get(3));
                builder.append("\"\n");
            } else if (function instanceof DynamicProgrammingNode) {
                for (int start: ((MetaList<Integer>) node.getAdditionalArguments().get(3)).getValues()) {
                    builder.append("\"");
                    builder.append(start);
                    builder.append("\",\n");
                }
                builder.deleteCharAt(builder.length() - 2);
            } else if (function instanceof StencilNode) {
                for (int j = 0; j < ((MetaList<Integer>) node.getAdditionalArguments().get(1)).getValues().size(); j++) {
                    long start = ((Number) ((MetaList<Integer>) node.getAdditionalArguments().get(1)).getValues().get(j)).longValue();

                    builder.append("\"");
                    builder.append(start);
                    builder.append("\", ");
                }
                builder.deleteCharAt(builder.length() - 2);
                builder.append(",\n");
                if (i + 1 == node.getAdditionalArguments().size() / node.getAdditionalArgumentCount()) {
                    builder.deleteCharAt(builder.length() - 2);
                }
            }
        }
        removeIndent();
        indent();
        builder.append("],\n");

        // Lengths
        indent();
        builder.append("\"Lengths\" : [\n");
        addIndent();
        for (int i = 0; i < node.getAdditionalArguments().size() / node.getAdditionalArgumentCount() ; i++) {
            FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
            indent();
            if (function instanceof MapNode) {
                builder.append("\"");
                builder.append(((MetaValue<Integer>) node.getAdditionalArguments().get(0)).getValue());
                builder.append("\"\n");
            } else if (function instanceof ReduceNode) {
                builder.append("\"");
                builder.append(((MetaList<Integer>) node.getAdditionalArguments().get(0)).getValues().get(0));
                builder.append("\"\n");
            } else if (function instanceof DynamicProgrammingNode) {
                builder.append("\"");
                builder.append(((MetaList<Integer>) node.getAdditionalArguments().get(0)).getValues().get(0));
                builder.append("\", ");
                builder.append("\"");
                builder.append(((MetaList<Integer>) node.getAdditionalArguments().get(0)).getValues().get(1));
                builder.append("\"\n");
            } else if (function instanceof StencilNode) {
                for (int j = 0; j < ((MetaList<Integer>) node.getAdditionalArguments().get(0)).getValues().size(); j++) {
                    long start = ((Number) ((MetaList<Integer>) node.getAdditionalArguments().get(0)).getValues().get(j)).longValue();
                    builder.append("\"");
                    builder.append(start);
                    builder.append("\", ");
                }
                builder.deleteCharAt(builder.length() - 2);
                builder.append(",\n");
                if (i + 1 == node.getAdditionalArguments().size() / node.getAdditionalArgumentCount()) {
                    builder.deleteCharAt(builder.length() - 2);
                }
            }
        }
        removeIndent();
        indent();
        builder.append("]\n");


    }

    /***************************************
     *
     *
     * Visitor necessities.
     *
     *
     ****************************************/
    private APT2JSON realThis = this;

    @Override
    public APT2JSON getRealThis() {
        return realThis;
    }

    public void setRealThis(APT2JSON realThis) {
        this.realThis = realThis;
    }
}
