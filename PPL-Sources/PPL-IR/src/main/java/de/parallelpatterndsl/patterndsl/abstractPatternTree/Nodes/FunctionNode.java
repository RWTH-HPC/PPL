package de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;

/**
 * Abstract definition for function definition nodes.
 */
public abstract class FunctionNode extends PatternNode {

    /**
     * The name of the function.
     */
    private String identifier;

    /**
     * The number of argument for a function call.
     */
    private int argumentCount;

    /**
     * by utilizing the extended shape Visitor this value can be used to access the current shape of parameters.
     */
    private int currentCall;

    /**
     * Stores the shapes of the parameters, for each time this function is called.
     * Only the shape for arrays will be stored, scalar values are ignored.
     * For parallel nodes: the last element of each set of parameters describes the shape of the return value.
     */
    private ArrayList<ArrayList<ArrayList<Integer>>> parameterShapes;

    /**
     * The Values given for a function call.
     */
    private ArrayList<Data> argumentValues = new ArrayList<>();

    /**
     * True, iff the function is still traversed after the inlining process.
     */
    private boolean availableAfterInlining = false;


    public FunctionNode(String identifier) {
        this.identifier = identifier;

        this.parameterShapes = new ArrayList<>();
        this.currentCall = 0;
    }

    public void setArgumentCount(int argumentCount) {
        this.argumentCount = argumentCount;
    }

    public void addArgumentValues(Data argumentValue) {
        this.argumentValues.add(argumentValue);
    }

    public String getIdentifier() {
        return identifier;
    }

    public int getArgumentCount() {
        return argumentCount;
    }

    public ArrayList<Data> getArgumentValues() {
        return argumentValues;
    }

    @Override
    public PatternNode getParent() {
        Log.error("Parent does not exist in function nodes!");
        throw new RuntimeException("Critical error!");
    }

    public int getCurrentCall() {
        return currentCall;
    }

    public void incrementCurrentCall() {
        this.currentCall++;
    }

    public void resetCurrentCall() {
        this.currentCall = 0;
    }

    public ArrayList<ArrayList<ArrayList<Integer>>> getParameterShapes() {
        return parameterShapes;
    }

    public void addParameterShapes(ArrayList<ArrayList<Integer>> shapes) {
        this.parameterShapes.add(shapes);
    }

    public boolean isAvailableAfterInlining() {
        return availableAfterInlining;
    }

    public void setAvailableAfterInlining(boolean availableAfterInlining) {
        this.availableAfterInlining = availableAfterInlining;
    }

    public abstract PrimitiveDataTypes getReturnType();

    public abstract boolean returnsArray();

    /**
     * Adds the identifier to the Function node, used when unrolling the APT
     * @param unrollIdentifier
     */
    public void addUnrollIdentifier(String unrollIdentifier) {
        identifier = identifier + "_" + unrollIdentifier;
    }

    @Override
    public abstract FunctionNode deepCopy();
}
