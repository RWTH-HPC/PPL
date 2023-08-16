package de.parallelpatterndsl.patterndsl.MappingTree.MemoryAllocator;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.GeneralDataPlacementFunctions.HandleDataPlacements;
import de.parallelpatterndsl.patterndsl.MappingTree.GeneralDataPlacementFunctions.OffloadDataEncoding;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.*;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.DynamicProgrammingMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.MainMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.ReduceMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.GPUParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ReductionCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Plain.ReturnMapping;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DynamicProgrammingDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.MapDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.StencilDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * This class defines and generates the (de-)allocation and data transfers for GPUs.
 */
public class MemoryAllocator {

    private MainMapping root;


    public MemoryAllocator(MainMapping root) {
        this.root = root;
    }

    public ArrayList<MappingNode> addAllocationNodes() {

        HashMap<Data, DataPlacement> dataPlacements = new HashMap<>();

        HashSet<OffloadDataEncoding> onGPU = new HashSet<>();

        ArrayList<MappingNode> result = new ArrayList<>();

        // initialize the original data places
        for (Data data : root.getVariableTable().values() ) {
            EndPoint endPoint;
            if (data instanceof ArrayData) {
                endPoint = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, ((ArrayData) data).getShape().get(0), new HashSet<>(), false);
            } else {
                endPoint = new EndPoint(AbstractMappingTree.getDefaultDevice(), 0, 1, new HashSet<>(), false);
            }

            ArrayList<EndPoint> initial = new ArrayList<>();
            initial.add(endPoint);

            DataPlacement placement = new DataPlacement(initial, data);
            dataPlacements.put(data, placement);
        }

        long duration = System.currentTimeMillis();
        // Handle (de-)allocations of data on the GPU
        for (int i = 0; i < root.getChildren().size(); i++) {
            if (i%200 == 0) {
                System.out.print("Completion " + i + " of " + root.getChildren().size() + ": ");
                System.out.println(((double) i)/root.getChildren().size() * 100);
                System.out.println("Took " + ((double) (System.currentTimeMillis() - duration)/1000) + "s");
            }

            MappingNode child = root.getChildren().get(i);

            // Handle allocation
            if (child instanceof ReductionCallMapping) {
                if (((ReductionCallMapping) child).isOnlyCombiner()) {
                    result.add(child);
                    continue;
                }
                if (((ReductionCallMapping) child).getOnGPU()) {
                    // Allocation of input data
                    if (((ReductionCallMapping) child).getTempInput().isEmpty()) {
                        for (DataPlacement necessary : child.getNecessaryData()) {
                            Optional<OffloadDataEncoding> present = HandleDataPlacements.isPresentOnGPU(necessary, onGPU);
                            if (present.isPresent()) {
                                ((ReductionCallMapping) child).getInputDataEncodings().add(present.get());
                                if (present.get().isMarkedForDeallocation()) {
                                    present.get().setMarkedForDeallocation(false);
                                    result = removeMarkedDeallocation(result, present.get());
                                }
                            } else {
                                if (necessary.getPlacement().size() > 1) {
                                    necessary = HandleDataPlacements.combineFracturedDataPlacements(necessary);
                                }
                                OffloadDataEncoding newEncoding = new OffloadDataEncoding(necessary, false, offsetFromMinShift((ParallelCallMapping) child, necessary.getDataElement()));
                                ((ReductionCallMapping) child).getInputDataEncodings().add(newEncoding);
                                onGPU.add(newEncoding);

                                // input Data allocation Mapping
                                GPUAllocationMapping allocationMapping = new GPUAllocationMapping(Optional.ofNullable(root), root.getVariableTable(), newEncoding, true);
                                result.add(allocationMapping);

                                // input Data transfer Mapping
                                GPUDataMovementMapping gpuDataMovementMapping = new GPUDataMovementMapping(Optional.of(root), root.getVariableTable(), newEncoding, true);
                                result.add(gpuDataMovementMapping);
                            }
                        }
                    }
                        // Allocation of output data
                        for (DataPlacement necessary: child.getOutputData() ) {
                            Optional<OffloadDataEncoding> present = HandleDataPlacements.isPresentOnGPU(necessary,onGPU);
                            if (present.isPresent()) {
                                ((ReductionCallMapping) child).getOutputDataEncodings().add(present.get());
                                present.get().setWriteAccessed(true);
                                if (present.get().isMarkedForDeallocation()) {
                                    present.get().setMarkedForDeallocation(false);
                                    result = removeMarkedDeallocation(result, present.get());
                                }
                            } else {
                                if (necessary.getPlacement().size() > 1) {
                                    necessary = HandleDataPlacements.combineFracturedDataPlacements(necessary);
                                }
                                OffloadDataEncoding newEncoding = new OffloadDataEncoding(necessary,true, offsetFromMinShift((ParallelCallMapping) child, necessary.getDataElement()));
                                ((ReductionCallMapping) child).getOutputDataEncodings().add(newEncoding);
                                if (newEncoding.getData() instanceof PrimitiveData && !((ReductionCallMapping) child).getTempOutput().isEmpty()) {
                                    // Mark reduction results, as they are combined differently
                                    newEncoding.setReductionResult(true);
                                }
                                onGPU.add(newEncoding);

                                // input Data allocation Mapping
                                GPUAllocationMapping allocationMapping = new GPUAllocationMapping(Optional.ofNullable(root), root.getVariableTable(), newEncoding, true);
                                result.add(allocationMapping);
                            }
                        }
                }
            } else
            if (child instanceof GPUParallelCallMapping) {
                // Allocation of input data
                for (DataPlacement necessary: child.getNecessaryData() ) {
                    Optional<OffloadDataEncoding> present = HandleDataPlacements.isPresentOnGPU(necessary,onGPU);
                    if (present.isPresent()) {
                        ((GPUParallelCallMapping) child).getInputDataEncodings().add(present.get());
                        if (present.get().isMarkedForDeallocation()) {
                            present.get().setMarkedForDeallocation(false);
                            result = removeMarkedDeallocation(result, present.get());
                        }
                    } else {
                        if (necessary.getPlacement().size() > 1) {
                            necessary = HandleDataPlacements.combineFracturedDataPlacements(necessary);
                        }
                        OffloadDataEncoding newEncoding = new OffloadDataEncoding(necessary, false, offsetFromMinShift((ParallelCallMapping) child, necessary.getDataElement()));
                        ((GPUParallelCallMapping) child).getInputDataEncodings().add(newEncoding);
                        onGPU.add(newEncoding);

                        // input Data allocation Mapping
                        GPUAllocationMapping allocationMapping = new GPUAllocationMapping(Optional.ofNullable(root), root.getVariableTable(), newEncoding, true);
                        result.add(allocationMapping);

                        // input Data transfer Mapping
                        GPUDataMovementMapping gpuDataMovementMapping = new GPUDataMovementMapping(Optional.of(root),root.getVariableTable(), newEncoding, true);
                        result.add(gpuDataMovementMapping);
                    }
                }

                // Allocation of output data
                for (DataPlacement necessary: child.getOutputData() ) {
                    Optional<OffloadDataEncoding> present = HandleDataPlacements.isPresentOnGPU(necessary,onGPU);
                    if (present.isPresent()) {
                        ((GPUParallelCallMapping) child).getOutputDataEncodings().add(present.get());
                        present.get().setWriteAccessed(true);
                        if (present.get().isMarkedForDeallocation()) {
                            present.get().setMarkedForDeallocation(false);
                            result = removeMarkedDeallocation(result, present.get());
                        }
                    } else {
                        if (necessary.getPlacement().size() > 1) {
                            necessary = HandleDataPlacements.combineFracturedDataPlacements(necessary);
                        }
                        OffloadDataEncoding newEncoding = new OffloadDataEncoding(necessary,true, offsetFromMinShift((ParallelCallMapping) child, necessary.getDataElement()));
                        ((GPUParallelCallMapping) child).getOutputDataEncodings().add(newEncoding);
                        onGPU.add(newEncoding);

                        // input Data allocation Mapping
                        GPUAllocationMapping allocationMapping = new GPUAllocationMapping(Optional.ofNullable(root), root.getVariableTable(), newEncoding, true);
                        result.add(allocationMapping);
                    }
                }
            } else {
                // Handle Dealloc
                if (child instanceof DataMovementMapping) {
                    HashSet<OffloadDataEncoding> overlap = HandleDataPlacements.hasGPUOverlap(onGPU, child.getNecessaryData());
                    for (OffloadDataEncoding encoding: overlap) {
                        handleGPUDataClosing(result, encoding);
                    }
                } else if (child instanceof BarrierMapping) {
                    HashSet<Device> sync = getDevices((BarrierMapping) child);
                    for (OffloadDataEncoding encoding: onGPU) {
                        if (sync.contains(encoding.getDevice())) {
                            handleGPUDataClosing(result, encoding);
                        }
                    }
                } else {
                    //Dealloc for read accesses
                    HashSet<OffloadDataEncoding> overlap = HandleDataPlacements.hasGPUOverlap(onGPU, child.getNecessaryData());
                    for (OffloadDataEncoding encoding: overlap) {
                        handleGPUDataClosing(result, encoding);
                    }

                    //Dealloc for write accesses
                    overlap = HandleDataPlacements.hasGPUOverlap(onGPU, child.getOutputData());
                    for (OffloadDataEncoding encoding: overlap) {
                        handleGPUDataClosing(result, encoding, onGPU);
                    }

                    //Handle Return nodes (dealloc all remaining data)
                    if (child instanceof ReturnMapping) {
                        for (OffloadDataEncoding encoding: onGPU ) {
                            if (!encoding.isMarkedForDeallocation()) {
                                result.add(new GPUDeAllocationMapping(Optional.of(root), root.getVariableTable(), encoding));
                            }
                        }
                    }

                }
            }



            result.add(child);

            /**
             * 1. DONE. Test if data is on GPU. If not create an alloc node. A GPU transfer node is only defined when the data is used as input.
             * 2. DONE. (OffloadDataEncoding). Add a unique identifier to the used data. This id must be reused by later calls, transfers and deallocations. Thus, it may not change
             * 3. DONE. If a subset of the data on the GPU output is used on a different device, copy back and mark for dealloc.
             * 4. DONE. On synchronization or MPI data transfer for the device/data, copy back all and mark for dealloc.
             * 5. DONE. On write access to allocated data mark for deallocation and confirm deallocation.
             * 6. DONE. When accessing the data on the same GPU again remove mark for dealloc.
             * (7.) Internode data transfers are assumed to be handled somewhere else before hand.
             */
        }


        // Handle Dynamic programming data transfers
        for (MappingNode child : root.getChildren() ) {
            if (child instanceof GPUParallelCallMapping) {
                if (AbstractMappingTree.getFunctionTable().get(((GPUParallelCallMapping) child).getFunctionIdentifier()) instanceof DynamicProgrammingMapping) {
                    for (OffloadDataEncoding input: ((GPUParallelCallMapping) child).getInputDataEncodings() ) {
                        ((GPUParallelCallMapping) child).addDpPostSwapTransfers(new GPUDataMovementMapping(Optional.of(child), child.getVariableTable(), input, true));
                    }
                    for (OffloadDataEncoding output: ((GPUParallelCallMapping) child).getOutputDataEncodings() ) {
                        ((GPUParallelCallMapping) child).addDpPreSwapTransfers(new GPUDataMovementMapping(Optional.of(child), child.getVariableTable(), output, false));
                    }
                }
            }
        }

        return result;
    }

    private void handleGPUDataClosing(ArrayList<MappingNode> result, OffloadDataEncoding encoding) {
        if (encoding.isWriteAccessed() && !encoding.isMarkedForDeallocation() && !encoding.isReductionResult()) {
            result.add(new GPUDataMovementMapping(Optional.of(root), root.getVariableTable(), encoding, false));
        }
        if (!encoding.isMarkedForDeallocation()) {
            result.add(new GPUDeAllocationMapping(Optional.of(root), root.getVariableTable(), encoding));
        }
        encoding.setMarkedForDeallocation(true);
    }

    private void handleGPUDataClosing(ArrayList<MappingNode> result, OffloadDataEncoding encoding, HashSet<OffloadDataEncoding> onGPU) {
        if (!encoding.isMarkedForDeallocation()) {
            result.add(new GPUDeAllocationMapping(Optional.of(root), root.getVariableTable(), encoding));
        }
        onGPU.remove(encoding);
    }
    /**
     * Returns the distance between the minimal shift and the shift of the current data element.
     * This is only necessary if the shift is < 0.
     * @param node
     * @param data
     * @return
     */
    private long offsetFromMinShift(ParallelCallMapping node, Data data) {
        if (!(data instanceof ArrayData))  {
            return 0;
        }
        FunctionMapping function = AbstractMappingTree.getFunctionTable().get(node.getFunctionIdentifier());
        ArrayList<ArrayData> plausibleInputs = new ArrayList<>();
        for (int i = 0; i < node.getArgumentExpressions().size(); i++) {
            OperationExpression argument = node.getArgumentExpressions().get(i);
            if (argument.getOperands().get(0) == data) {
                if (function.getArgumentValues().get(i) instanceof ArrayData) {
                    plausibleInputs.add((ArrayData) function.getArgumentValues().get(i));
                }
            }
        }

        long minOffset = 0;
        long globalMinOffset = 0;
        ArrayList<DataAccess> allAccesses = new ArrayList<>();
        allAccesses.addAll(function.getInputAccesses());
        allAccesses.addAll(function.getOutputAccesses());
        for (DataAccess access: allAccesses) {
            if (!(access.getData() instanceof ArrayData)) {
                continue;
            }
            if (access instanceof MapDataAccess) {
                globalMinOffset = Long.min(globalMinOffset, ((MapDataAccess) access).getShiftOffset());
                if (plausibleInputs.contains((ArrayData) access.getData())) {
                    minOffset = Long.min(minOffset, ((MapDataAccess) access).getShiftOffset());
                }
            } else if (access instanceof StencilDataAccess) {
                if (((StencilDataAccess) access).getRuleBaseIndex().get(0).equals("INDEX0")) {
                    globalMinOffset = Long.min(globalMinOffset, ((StencilDataAccess) access).getShiftOffsets().get(0));
                    if (plausibleInputs.contains((ArrayData) access.getData())) {
                        minOffset = Long.min(minOffset, ((StencilDataAccess) access).getShiftOffsets().get(0));
                    }
                }
            } else if (access instanceof DynamicProgrammingDataAccess) {
                globalMinOffset = Long.min(globalMinOffset, ((DynamicProgrammingDataAccess) access).getShiftOffsets().get(0));
                if (plausibleInputs.contains((ArrayData) access.getData())) {
                    minOffset = Long.min(minOffset, ((DynamicProgrammingDataAccess) access).getShiftOffsets().get(0));
                }
            }
        }
        return -1*(globalMinOffset - minOffset);
    }

    /**
     * Removes a deallocation from a set of children.
     * @param scope
     * @param toRemove
     * @return
     */
    private ArrayList<MappingNode> removeMarkedDeallocation(ArrayList<MappingNode> scope, OffloadDataEncoding toRemove) {
        return new ArrayList<>(scope.stream().filter(x -> testForRemoval(x, toRemove)).collect(Collectors.toList()));
    }

    private boolean testForRemoval(MappingNode toTest, OffloadDataEncoding toRemove) {
        if (toTest instanceof GPUDeAllocationMapping) {
            if (((GPUDeAllocationMapping) toTest).getAllocator() == toRemove) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns the devices synchronized by the barrier.
     * @param node
     * @return
     */
    private HashSet<Device> getDevices(BarrierMapping node) {
        HashSet<Device> result = new HashSet<>();
        for (Processor processor:node.getBarrierProc() ) {
            result.add(processor.getParent());
        }
        return result;
    }


}
