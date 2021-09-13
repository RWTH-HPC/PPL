package de.parallelpatterndsl.patterndsl.MappingTree.Nodes;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.Optional;

/**
 * Definition of a function within the abstract mapping tree.
 */
public abstract class FunctionMapping extends MappingNode {

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
     * True, iff this function does have a parallel pattern node as one of its ancestors.
     */
    private boolean hasParallelDescendants;


    public FunctionMapping(FunctionNode aptNode){
        super(Optional.empty(), aptNode.getVariableTable(), aptNode);
        this.identifier = aptNode.getIdentifier();
        this.argumentCount = aptNode.getArgumentCount();
        this.argumentValues = aptNode.getArgumentValues();
        this.currentCall = 0;
        this.hasParallelDescendants = aptNode.isHasParallelDescendants();
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

    public boolean isHasParallelDescendants() {
        return hasParallelDescendants;
    }

    @Override
    public MappingNode getParent() {
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
}
