package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.CombinerFunction;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.ReduceNode;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Optional;

/**
 * Defines a reduction in an abstract mapping tree.
 */
public class ReduceMapping extends ParallelMapping {

    /**
     * the combiner function used in the reduction.
     */
    private CombinerFunction combinerFunction;

    public ReduceMapping(ReduceNode aptNode) {
        super(aptNode);
        combinerFunction = aptNode.getCombinerFunction();
    }


    public CombinerFunction getCombinerFunction() {
        return combinerFunction;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
