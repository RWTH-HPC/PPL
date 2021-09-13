package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.StencilNode;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Optional;

/**
 * Defines a stencil in the abstract mapping tree.
 */
public class StencilMapping extends ParallelMapping {

    /**
     * The dimensionality of the underlying data structure.
     */
    private int dimension;

    public StencilMapping(StencilNode aptNode) {
        super(aptNode);
        this.dimension = aptNode.getDimension();
    }

    public int getDimension() {
        return dimension;
    }



    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
