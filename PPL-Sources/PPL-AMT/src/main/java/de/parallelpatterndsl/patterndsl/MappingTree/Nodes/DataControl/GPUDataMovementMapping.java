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
public class GPUDataMovementMapping extends AbstractDataMovementMapping {



    /**
     * Stores where the data is allocated.
     */
    private OffloadDataEncoding Allocator;

    /**
     * True, iff the GPU data defines an input.
     */
    private boolean IsCPU2GPU;


    public GPUDataMovementMapping(Optional<MappingNode> parent, HashMap<String, Data> variableTable, OffloadDataEncoding allocator, boolean isCPU2GPU) {
        super(parent, variableTable, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        Allocator = allocator;
        IsCPU2GPU = isCPU2GPU;
        children = new ArrayList<>();
    }

    public OffloadDataEncoding getAllocator() {
        return Allocator;
    }

    public boolean isCPU2GPU() {
        return IsCPU2GPU;
    }


    @Override
    public HashSet<DataPlacement> getNecessaryData() {
        // The data to be allocated should be present on the node
        HashSet<DataPlacement> result = new HashSet<>();
        if (IsCPU2GPU) {
            ArrayList<EndPoint> endPoints = new ArrayList<>();
            for (EndPoint endPoint : Allocator.getDataPlacement().getPlacement()) {
                EndPoint targetPoint = new EndPoint(endPoint.getLocation().getParent().getDevices().get(0), endPoint.getStart(), endPoint.getLength(), endPoint.getParallelAccess(), endPoint.isHasParallelWriteAccess());
                endPoints.add(targetPoint);
            }
            DataPlacement targetPlacement = new DataPlacement(endPoints, Allocator.getDataPlacement().getDataElement());
            result.add(targetPlacement);
        } else {
            result.add(Allocator.getDataPlacement());
        }
        return result;
    }

    @Override
    public HashSet<DataPlacement> getOutputData() {
        HashSet<DataPlacement> result = new HashSet<>();
        if (IsCPU2GPU) {
            result.add(Allocator.getDataPlacement());
        } else {
            ArrayList<EndPoint> endPoints = new ArrayList<>();
            for (EndPoint endPoint : Allocator.getDataPlacement().getPlacement()) {
                EndPoint targetPoint = new EndPoint(endPoint.getLocation().getParent().getDevices().get(0), endPoint.getStart(), endPoint.getLength(), endPoint.getParallelAccess(), endPoint.isHasParallelWriteAccess());
                endPoints.add(targetPoint);
            }
            DataPlacement targetPlacement = new DataPlacement(endPoints, Allocator.getDataPlacement().getDataElement());
            result.add(targetPlacement);
        }
        return result;
    }

    /**
     * Visitor accept function.
     */
    public void accept(AMTVisitor visitor) {
        visitor.handle(this);
    }
}
