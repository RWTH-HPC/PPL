package de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.BarrierMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataMovementMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.DynamicProgrammingMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.MainMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.MappingNode;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.FusedParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.ParallelCalls.ParallelCallMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.SupportFunction;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;

import java.util.*;

/**
 * This class implements an execution model to generate appropriate synchronization and data transfer structures.
 */
public class AbstractSynchronizationModel {

    /**
     * Variable describing the root node of the AMT
     */
    private MainMapping mainMapping;

    /**
     * Variable describing the target architecture
     */
    private Network network;

    /**
     * A map containing the parallel groups accessible by the function inline data element.
     * For fused parallel calls the function inline data of the first parallel call in the pipeline is used.
     */
    private HashMap<FunctionInlineData, ParallelGroup> parallelCallGroups;

    public AbstractSynchronizationModel(MainMapping mainMapping, Network network) {
        this.mainMapping = mainMapping;
        this.network = network;
    }

    /**
     * This function generates all synchronization and data transfer nodes necessary for the code to avoid data races.
     * @return The result is a set of extended child nodes for the root of the AMT.
     */
    public ArrayList<MappingNode> generateSyncAndTransfer() {
        HashMap<Data, ArrayList<EndPoint>> initialDataPlacement = new HashMap<>();

        HashMap<Data, DataPlacement> dataPlacements = new HashMap<>();

        ArrayList<MappingNode> extendedParallelCalls = new ArrayList<>();

        generateParallelCallGroups();

        // connect parallel call groups
        for (MappingNode child: mainMapping.getChildren() ) {
            if (child instanceof ParallelCallMapping) {
                CallGroup callGroup = (CallGroup) parallelCallGroups.get(((ParallelCallMapping) child).getCallExpression());
                if (callGroup.isFirstAccess() && callGroup.getParameterReplacementExpressions().isPresent()) {
                    extendedParallelCalls.add(callGroup.getParameterReplacementExpressions().get());
                }

                extendedParallelCalls.add(child);
                ((ParallelCallMapping) child).setGroup(callGroup);

                if (callGroup.isLastCall() && callGroup.getResultReplacementExpression().isPresent()) {
                    extendedParallelCalls.add(callGroup.getResultReplacementExpression().get());
                }

            } else if (child instanceof FusedParallelCallMapping) {
                FusedCallGroup callGroup = (FusedCallGroup) parallelCallGroups.get(((ParallelCallMapping) child.getChildren().get(0)).getCallExpression());
                if (callGroup.isFirstAccess() && callGroup.getParameterReplacementExpressions().isPresent()) {
                    extendedParallelCalls.add(callGroup.getParameterReplacementExpressions().get());
                }

                extendedParallelCalls.add(child);
                for (int i = 0; i < child.getChildren().size(); i++) {
                    ((ParallelCallMapping) child.getChildren().get(i)).setGroup(callGroup);
                }


                if (callGroup.isLastCall() && callGroup.getResultReplacementExpression().isPresent()) {
                    extendedParallelCalls.add(callGroup.getResultReplacementExpression().get());
                }

            } else {
                extendedParallelCalls.add(child);
            }
        }

        // create initial data placements
        for (Data vars : mainMapping.getVariableTable().values() ) {
            ArrayList<EndPoint> setUp = new ArrayList<>();

            EndPoint initial;

            if (vars instanceof PrimitiveData) {
                initial = new EndPoint(network.getNodes().get(0).getDevices().get(0), 0, 1, new HashSet<>(), false);
                setUp.add(initial);
            } else if (vars instanceof ArrayData) {
                initial = new EndPoint(network.getNodes().get(0).getDevices().get(0), 0,  ((ArrayData) vars).getShape().get(0), new HashSet<>(), false);
                setUp.add(initial);
            }

            initialDataPlacement.put(vars, setUp);

            for (Data data: initialDataPlacement.keySet() ) {
                DataPlacement placement = new DataPlacement(initialDataPlacement.get(data), data);
                dataPlacements.put(data, placement);
            }

        }


        for (ParallelGroup group: parallelCallGroups.values() ) {
            group.setFirstAccess(true);
            group.resetRemaining();
        }

        ArrayList<MappingNode> synchronizedNodes = new ArrayList<>();

        for (MappingNode child: extendedParallelCalls ) {
            HashSet<DataPlacement> inputData = child.getNecessaryData();
            HashSet<DataPlacement> outputData = child.getOutputData();

            // generate data transfers and synchronization for dynamic programming recursions.
            if (child instanceof ParallelCallMapping) {
                FunctionMapping functionMapping =  AbstractMappingTree.getFunctionTable().get(((ParallelCallMapping) child).getFunctionIdentifier());
                if (functionMapping instanceof DynamicProgrammingMapping) {
                    createDPSync(child.getParent(), (ParallelCallMapping) child);
                }
            }

            Optional<BarrierMapping> barrierMapping = Optional.empty();




            Optional<ParallelGroup> groupOPT = Optional.empty();
            if (child instanceof ParallelCallMapping) {
                ParallelGroup group = parallelCallGroups.get(((ParallelCallMapping) child).getCallExpression());
                if (group.isLastCall()) {
                    barrierMapping = generateSynchronization(dataPlacements, inputData, outputData);
                }
                groupOPT = Optional.of(group);
            } else if (child instanceof FusedParallelCallMapping) {
                ParallelGroup group = parallelCallGroups.get(((ParallelCallMapping) child.getChildren().get(0)).getCallExpression());
                groupOPT = Optional.of(group);
                if (group.isLastCall()) {
                    barrierMapping = generateSynchronization(dataPlacements, inputData, outputData);
                }
            }




            if (groupOPT.isPresent()) {
                ParallelGroup group = groupOPT.get();
                if (group.isFirstAccess()) {
                    group.setFirstAccess(false);
                    for (ArrayList<DataPlacement> placements : group.getFullInputPlacement().values()) {
                        ArrayList<EndPoint> combined = new ArrayList<>();
                        for (DataPlacement placement: placements ) {
                            combined.addAll(placement.getPlacement());
                        }
                        Optional<DataMovementMapping> dataTransfer = generateDataMovement(mainMapping, dataPlacements.get(placements.get(0).getDataElement()), new DataPlacement(combined,placements.get(0).getDataElement()));

                        dataTransfer.ifPresent(synchronizedNodes::add);
                    }
                }
            } else {
                for (DataPlacement placement : child.getNecessaryData()) {
                    Optional<DataMovementMapping> dataTransfer = generateDataMovement(mainMapping, dataPlacements.get(placement.getDataElement()), placement);

                    dataTransfer.ifPresent(synchronizedNodes::add);
                }
            }

            synchronizedNodes.add(child);
            barrierMapping.ifPresent(synchronizedNodes::add);

            // Handle changed data placements
            for (DataPlacement readAccess: inputData) {
                DataPlacement statusQuo = dataPlacements.get(readAccess.getDataElement());
                dataPlacements.remove(readAccess.getDataElement());
                dataPlacements.put(readAccess.getDataElement(), createReadPlacements(statusQuo, readAccess));
            }

            for (DataPlacement writeAccess: outputData) {
                DataPlacement statusQuo = dataPlacements.get(writeAccess.getDataElement());
                dataPlacements.remove(writeAccess.getDataElement());
                dataPlacements.put(writeAccess.getDataElement(), createWritePlacement(statusQuo, writeAccess));
            }
        }


        for (ParallelGroup group: parallelCallGroups.values() ) {
            group.setFirstAccess(true);
            group.resetRemaining();
        }

        return synchronizedNodes;
    }

    /**
     * Combines the original data placement with the data placement which is being read and returns it.
     * @param original
     * @param read
     * @return
     */
    private DataPlacement createReadPlacements(DataPlacement original, DataPlacement read) {

        ArrayList<EndPoint> newEndPoints = original.getPlacement();
        newEndPoints.addAll(read.getPlacement());

        return new DataPlacement(newEndPoints, original.getDataElement());
    }

    /**
     * Combines the original data placement with the data placement which is being written to and returns it.
     * @param original
     * @param write
     * @return
     */
    private DataPlacement createWritePlacement(DataPlacement original, DataPlacement write) {
        long start = 0;

        long globalLength = 1;

        if (original.getDataElement() instanceof ArrayData) {
            globalLength = ((ArrayData) original.getDataElement()).getShape().get(0);
        }

        ArrayList<EndPoint> combinedPlacements = new ArrayList<>();

        // generate end points for each changed end point and its predecessors.
        for (int i = 0; i < write.getPlacement().size(); i++) {
            EndPoint writePlacement = write.getPlacement().get(i);
            if (writePlacement.getStart() != start) {
                for (EndPoint predecessor: original.getPlacement()) {
                    if (predecessor.getStart() < writePlacement.getStart()) {
                        if (predecessor.getStart() + predecessor.getLength() >= writePlacement.getStart()) {
                            long startIndex = SupportFunction.max(start, predecessor.getStart());
                            long lengthIndex = SupportFunction.min(writePlacement.getStart() - startIndex, predecessor.getLength());
                            EndPoint split = new EndPoint(predecessor.getLocation(), startIndex,lengthIndex , predecessor.getParallelAccess(), predecessor.isHasParallelWriteAccess() );
                            combinedPlacements.add(split);
                            start = startIndex + lengthIndex;
                            globalLength = globalLength - lengthIndex;
                        } else {
                            combinedPlacements.add(predecessor);
                            start = predecessor.getStart() + predecessor.getLength();
                            globalLength = globalLength - predecessor.getLength();
                        }
                    } else {
                        break;
                    }
                }
            }
            combinedPlacements.add(writePlacement);
            start = writePlacement.getStart() + writePlacement.getLength();
            globalLength = globalLength - writePlacement.getLength();
        }

        // generate the remaining end points after the last write access.
        if (start < globalLength) {
            for (EndPoint successor: original.getPlacement()) {
                if (successor.getStart() > start) {
                    long startIndex = SupportFunction.max(start, successor.getStart());
                    long lengthIndex = SupportFunction.min(globalLength - startIndex, successor.getLength());
                    EndPoint split = new EndPoint(successor.getLocation(), startIndex,lengthIndex , successor.getParallelAccess(), successor.isHasParallelWriteAccess() );
                    combinedPlacements.add(split);
                }
            }
        }
        return new DataPlacement(combinedPlacements, write.getDataElement());
    }

    /**
     * This function generates an applicable DataMovementMapping given the original Data placement and the input placement for the next Mapping node.
     * @param parent
     * @param original
     * @param input
     * @return
     */
    private Optional<DataMovementMapping> generateDataMovement(MappingNode parent, DataPlacement original, DataPlacement input) {
        if (input.getPlacement().isEmpty()) {
            return Optional.empty();
        }

        ArrayList<EndPoint> reducedInput = removeOverlaps(original, input);

        if (reducedInput.isEmpty()) {
            return Optional.empty();
        }


        ArrayList<EndPoint> sourcePoints = getSourceData(original, reducedInput);

        DataPlacement destination = new DataPlacement(reducedInput, input.getDataElement());
        HashSet<DataPlacement> destSet= new HashSet<>();
        destSet.add(destination);

        DataPlacement source = new DataPlacement(sourcePoints, original.getDataElement());
        HashSet<DataPlacement> sourceSet = new HashSet<>();
        sourceSet.add(source);

        DataMovementMapping result = new DataMovementMapping(Optional.of(parent), parent.getVariableTable(), sourceSet, destSet);


        return Optional.of(result);
    }

    /**
     * Returns a list of EndPoints suitable as a source for the data communication.
     * @param original
     * @param destination
     * @return
     */
    private ArrayList<EndPoint> getSourceData(DataPlacement original, ArrayList<EndPoint> destination) {
        ArrayList<EndPoint> result = new ArrayList<>();
        for (EndPoint target: destination ) {
            result.addAll(getIndividualSource(original, target));
        }
        return result;
    }

    /**
     * Returns a list of EndPoints covering all data elements in target.
     * Currently the algorithm heavily relies on the SORTED list of endpoints in a data placement and implements a first comes first serves approach.
     * Thus, the result may not be optimal.
     * @param original
     * @param target
     * @return
     */
    private ArrayList<EndPoint> getIndividualSource(DataPlacement original, EndPoint target) {
        long currentStart = target.getStart();
        long currentLength = target.getLength();
        ArrayList<EndPoint> result = new ArrayList<>();
        for (EndPoint currentSource: original.getPlacement()) {

            if (currentSource.getStart() <= currentStart && currentSource.getStart() + currentSource.getLength() > currentStart) {
                if (currentSource.getStart() + currentSource.getLength() >= currentStart + currentLength) {
                    EndPoint source = new EndPoint(currentSource.getLocation(), currentStart, currentLength, currentSource.getParallelAccess(), currentSource.isHasParallelWriteAccess());
                    result.add(source);
                    break;
                } else {
                    long length = currentSource.getStart() + currentSource.getLength() - currentStart;
                    EndPoint source = new EndPoint(currentSource.getLocation(), currentStart, length, currentSource.getParallelAccess(), currentSource.isHasParallelWriteAccess());
                    result.add(source);
                    currentStart = currentStart + length;
                    currentLength = currentLength - length;
                }
            }
        }
        return result;
    }

    /**
     * Tests, if the dataplacement input overlaps with the original on the same device and reduces the number of targets for the data transfer as a result.
     * @param original
     * @param input
     * @return
     */
    private ArrayList<EndPoint> removeOverlaps( DataPlacement original, DataPlacement input) {
        ArrayList<EndPoint> result = new ArrayList<>();
        for (EndPoint destination: input.getPlacement() ) {
            // avoid redundant data movements, by testing for overlaps with a single Endpoint and choosing the smallest destination
            // Partial overlaps can be reduced into even smaller chunks, which is not done here for simplicity.
            ArrayList<ArrayList<EndPoint>> reductionCandidates = new ArrayList<>();
            for (EndPoint source: original.getPlacement() ) {
                if (source.getLocation() == destination.getLocation()) {
                    if (source.getStart() <= destination.getStart() && source.getLength() >= destination.getLength()) {
                        reductionCandidates.add(new ArrayList<>());
                        break;
                    } else if (source.getStart() <= destination.getStart() && source.getLength() + source.getStart() < destination.getLength() + destination.getStart()) {
                        long newStart = source.getStart() + source.getLength();
                        long newLength = (destination.getLength() + destination.getStart()) - (source.getLength() + source.getStart());
                        EndPoint reducedDestination = new EndPoint(destination.getLocation(), newStart, newLength, destination.getParallelAccess(), destination.isHasParallelWriteAccess() );
                        ArrayList<EndPoint> partial = new ArrayList<>();
                        partial.add(reducedDestination);
                        reductionCandidates.add(partial);
                    } else if( source.getStart() > destination.getStart() && source.getLength() + source.getStart() >= destination.getLength() + destination.getStart()) {
                        long newLength = (source.getLength() + source.getStart()) - (destination.getLength() + destination.getStart());
                        EndPoint reducedDestination = new EndPoint(destination.getLocation(), destination.getStart(), newLength, destination.getParallelAccess(), destination.isHasParallelWriteAccess());
                        ArrayList<EndPoint> partial = new ArrayList<>();
                        partial.add(reducedDestination);
                        reductionCandidates.add(partial);
                    } else if (source.getStart() > destination.getStart() && source.getLength() + source.getStart() < destination.getLength() + destination.getStart()) {
                        long firstStart = destination.getStart();
                        long firstLength = source.getStart() - destination.getStart();
                        long secoundStart = source.getStart() + source.getLength();
                        long secondLength = destination.getStart() + destination.getLength() - secoundStart;
                        EndPoint first = new EndPoint(destination.getLocation(), firstStart, firstLength, destination.getParallelAccess(), destination.isHasParallelWriteAccess());
                        EndPoint second = new EndPoint(destination.getLocation(), secoundStart, secondLength, destination.getParallelAccess(), destination.isHasParallelWriteAccess());
                        ArrayList<EndPoint> partial = new ArrayList<>();
                        partial.add(first);
                        partial.add(second);
                        reductionCandidates.add(partial);
                    }
                }
            }
            ArrayList<EndPoint> partial = new ArrayList<>();
            partial.add(destination);
            reductionCandidates.add(partial);

            result.addAll(getSmallestOverlap(reductionCandidates));
        }
        return result;
    }

    /**
     * Chooses the minimal set of Endpoints from multiple overlapping sets and returns it.
     * @param inputChoices
     * @return
     */
    private ArrayList<EndPoint> getSmallestOverlap(ArrayList<ArrayList<EndPoint>> inputChoices) {
        ArrayList<EndPoint> minimum = new ArrayList<>();
        int minSize = Integer.MAX_VALUE;

        for (ArrayList<EndPoint> choice: inputChoices ) {
            int currentSize = 0;
            for (EndPoint element: choice ) {
                currentSize += element.getLength();
            }
            if (currentSize < minSize) {
                minSize = currentSize;
                minimum = choice;
            }
        }
        return minimum;
    }


    /**
     * generates a Barrier node iff applicable, based on the current data placement (statusQuo) and the input and output data placements of the current node.
     * @param statusQuo
     * @param input
     * @param output
     * @return
     */
    private Optional<BarrierMapping> generateSynchronization(HashMap<Data, DataPlacement> statusQuo, HashSet<DataPlacement> input, HashSet<DataPlacement> output) {
        HashSet<ParallelGroup> groups = new HashSet<>();
        for (DataPlacement inputData: input ) {
            groups.addAll(getSyncTargets(statusQuo.get(inputData.getDataElement()), inputData));
        }
        for (DataPlacement outputData: output ) {
            groups.addAll(getSyncTargets(statusQuo.get(outputData.getDataElement()), outputData));
        }

        if (groups.isEmpty()) {
            return Optional.empty();
        }

        BarrierMapping barrier = new BarrierMapping(Optional.of(mainMapping), mainMapping.getVariableTable(), groups );
        return Optional.of(barrier);
    }

    /**
     * Returns all parallel groups that need to be synchronized in order to avoid data races with the next step.
     * @param present
     * @param newPlacement
     * @return
     */
    private Set<ParallelGroup> getSyncTargets(DataPlacement present, DataPlacement newPlacement) {
        HashSet<ParallelGroup> result = new HashSet<>();
        // If applicable store which parts are accessed in parallel and not only if.
        boolean hasParallelAccess = false;
        for (EndPoint source: getOverlap(present, newPlacement) ) {
            if (source.isHasParallelWriteAccess()) {
                result.addAll(source.getParallelAccess());
                source.setHasParallelWriteAccess(false);
            } else if (!source.getParallelAccess().isEmpty()) {
                hasParallelAccess = true;
            }
        }
        for (EndPoint target: newPlacement.getPlacement() ) {
            if (target.isHasParallelWriteAccess() && hasParallelAccess) {
                result.addAll(target.getParallelAccess());
            }
        }

        return result;
    }

    /**
     * Returns a set of endpoints from the present data placement. These endpoints all overlap with endpoints in the newDataplacement.
     * @param present
     * @param newPlacement
     * @return
     */
    private HashSet<EndPoint> getOverlap(DataPlacement present, DataPlacement newPlacement) {
        HashSet<EndPoint> result = new HashSet<>();
        for (EndPoint target: newPlacement.getPlacement() ) {
            for (EndPoint original: present.getPlacement()) {
                if (doOverlap(original, target)) {
                    result.add(target);
                }
            }
        }
        return result;
    }

    /**
     * Returns true, iff the two endpoints overlap in their data elements.
     * @param first
     * @param second
     * @return
     */
    private boolean doOverlap(EndPoint first, EndPoint second) {
        if (first.getStart() <= second.getStart() && first.getStart() + first.getLength() >= second.getStart()) {
            return true;
        } else if (second.getStart() <= first.getStart() && second.getStart() + second.getLength() >= first.getStart()){
            return true;
        } else {
            return false;
        }
    }

    /**
     * Generates the barrier and data transfer for dynamic programming nodes.
     * @param parent
     * @param node
     */
    private void createDPSync(MappingNode parent, ParallelCallMapping node) {

        DynamicProgrammingMapping functionMapping = (DynamicProgrammingMapping) AbstractMappingTree.getFunctionTable().get(node.getFunctionIdentifier());

        HashSet<ParallelGroup> barrier = new HashSet<>();

        ParallelGroup group = parallelCallGroups.get(node.getCallExpression());

        barrier.add(group);

        BarrierMapping dpBarrier = new BarrierMapping(Optional.of(parent), node.getVariableTable(), barrier);

        HashSet<DataMovementMapping> dataMovementMappings = new HashSet<>();


        ArrayList<ParallelCallMapping> calls = new ArrayList<>();

        if (group instanceof CallGroup) {
            calls.addAll(((CallGroup)group).getGroup());
        } else if ( group instanceof FusedCallGroup) {
            for (FusedParallelCallMapping call: ((FusedCallGroup)group).getGroup() ) {
                calls.add((ParallelCallMapping) call.getChildren().get(0));
            }
        }

        for (int i = 0; i < calls.size(); i++) {
            HashSet<DataPlacement> targets = new HashSet<>(calls.get(i).getOutputData());
            HashSet<DataPlacement> destination = new HashSet<>();
            for (int j = 0; j < calls.size(); j++) {
                if (i == j) {
                    continue;
                }
                //Recreate the necessary data slice on a different device as the destination
                HashSet<DataPlacement> subDestination =  new HashSet<>(targets);
                for (DataPlacement placement: subDestination) {
                    ArrayList<EndPoint> endPoints = new ArrayList<>();
                    for (EndPoint endPoint: placement.getPlacement() ) {
                        EndPoint copy = endPoint.clone();
                        copy.setLocation(calls.get(j).getExecutor().getParent());
                        endPoints.add(copy);
                    }
                    destination.add(new DataPlacement(endPoints, calls.get(j).getOutputElements().get(0)));
                }
            }
            dataMovementMappings.add(new DataMovementMapping(Optional.of(parent), node.getVariableTable(), targets, destination));
        }

        node.setDynamicProgrammingBarrier(Optional.of(dpBarrier));
        node.setDynamicProgrammingdataTransfers(dataMovementMappings);

    }


    /**
     * This function groups the same (fused-) parallel calls and the number of splits.
     */
    private void generateParallelCallGroups() {
        HashMap<FunctionInlineData, ArrayList<ParallelCallMapping>> localParallelCallGroups = new HashMap<>();
        parallelCallGroups = new HashMap<>();

        HashMap<FunctionInlineData, ArrayList<FusedParallelCallMapping>> localFusedParallelCallGroups = new HashMap<>();

        for (MappingNode child: mainMapping.getChildren() ) {
            if (child instanceof ParallelCallMapping) {
                if (!localParallelCallGroups.containsKey(((ParallelCallMapping) child).getCallExpression())) {
                    ArrayList<ParallelCallMapping> initial = new ArrayList<>();
                    initial.add((ParallelCallMapping) child);
                    localParallelCallGroups.put(((ParallelCallMapping) child).getCallExpression(), initial);
                } else {
                    localParallelCallGroups.get(((ParallelCallMapping) child).getCallExpression()).add((ParallelCallMapping) child);
                }
            } else if (child instanceof FusedParallelCallMapping) {
                if (!localFusedParallelCallGroups.containsKey(((ParallelCallMapping) child.getChildren().get(0)).getCallExpression())) {
                    ArrayList<FusedParallelCallMapping> initial = new ArrayList<>();
                    initial.add((FusedParallelCallMapping) child);
                    localFusedParallelCallGroups.put(((ParallelCallMapping) child.getChildren().get(0)).getCallExpression(), initial);
                } else {
                    localFusedParallelCallGroups.get(((ParallelCallMapping) child.getChildren().get(0)).getCallExpression()).add((FusedParallelCallMapping) child);
                }

                // add dynamic programming nodes
                for (MappingNode grandChild: child.getChildren() ) {
                    if (grandChild instanceof ParallelCallMapping) {
                        if (AbstractMappingTree.getFunctionTable().get(((ParallelCallMapping) grandChild).getFunctionIdentifier()) instanceof DynamicProgrammingMapping) {
                            if (!localParallelCallGroups.containsKey(((ParallelCallMapping) grandChild).getCallExpression()) || !localFusedParallelCallGroups.containsKey(((ParallelCallMapping) grandChild.getChildren().get(0)).getCallExpression())) {
                                ArrayList<ParallelCallMapping> initial = new ArrayList<>();
                                initial.add((ParallelCallMapping) grandChild);
                                localParallelCallGroups.put(((ParallelCallMapping) grandChild).getCallExpression(), initial);
                            } else {
                                localParallelCallGroups.get(((ParallelCallMapping) grandChild).getCallExpression()).add((ParallelCallMapping) grandChild);
                            }
                        }
                    }
                }
            }
        }

        for (FunctionInlineData key : localParallelCallGroups.keySet() ) {
            parallelCallGroups.put(key, new CallGroup(mainMapping, localParallelCallGroups.get(key)));
        }

        for (FunctionInlineData key : localFusedParallelCallGroups.keySet() ) {
            parallelCallGroups.put(key, new FusedCallGroup(mainMapping, localFusedParallelCallGroups.get(key)));
        }
    }

}
