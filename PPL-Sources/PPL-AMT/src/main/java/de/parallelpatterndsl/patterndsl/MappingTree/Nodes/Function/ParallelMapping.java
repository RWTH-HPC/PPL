package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.ParallelNode;

import java.util.ArrayList;

/**
 * Class defining a parallel function in the abstract mapping tree.
 */
public abstract class ParallelMapping extends FunctionMapping {



    /**
     * The Data element, which will be returned at the end of parallel node.
     */
    private Data returnElement;

    public ParallelMapping(ParallelNode aptNode) {
        super(aptNode);
        this.returnElement = aptNode.getReturnElement();
    }


    public Data getReturnElement() {
        return returnElement;
    }
}
