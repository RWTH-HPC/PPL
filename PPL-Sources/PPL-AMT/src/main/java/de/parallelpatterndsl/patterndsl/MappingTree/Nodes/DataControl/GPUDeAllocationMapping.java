package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl;

import de.parallelpatterndsl.patterndsl.MappingTree.GeneralDataPlacementFunctions.OffloadDataEncoding;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Visitor.AMTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;

/**
 * Class defining data movement in an abstract mapping tree.
 */
public class GPUDeAllocationMapping extends MappingNode {



    /**
     * Stores where the data is allocated.
     */
    private OffloadDataEncoding Allocator;


    public GPUDeAllocationMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, OffloadDataEncoding allocator) {
        super(parent, variableTable, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        Allocator = allocator;
        children = new ArrayList<>();
    }

    public OffloadDataEncoding getAllocator() {
        return Allocator;
    }

    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        // The data to be allocated should be present on the node
        HashSet<DataPlacement> result = new HashSet<>();
        result.add(Allocator.getDataPlacement());
        return result;
    }

    @Override
    public HashSet<DataPlacement> getOutputData() {
        return  new HashSet<>();
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
