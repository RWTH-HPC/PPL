package de.parallelpatterndsl.patterndsl.MappingTree.GeneralDataPlacementFunctions;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.DataPlacement;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl.EndPoint;
import de.parallelpatterndsl.patterndsl.MappingTree.SupportFunction;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Optional;

public class HandleDataPlacements {

    /**
     * Tests, if the dataplacement input overlaps with the original on the same device and reduces the number of targets for the data transfer as a result.
     * @param original
     * @param input
     * @return
     */
    public static ArrayList<EndPoint> removeOverlaps(DataPlacement original, DataPlacement input) {
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
    private static ArrayList<EndPoint> getSmallestOverlap(ArrayList<ArrayList<EndPoint>> inputChoices) {
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
     * Returns a list of EndPoints suitable as a source for the data communication.
     * @param original
     * @param destination
     * @return
     */
    public static ArrayList<EndPoint> getSourceData(DataPlacement original, ArrayList<EndPoint> destination) {
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
    private static ArrayList<EndPoint> getIndividualSource(DataPlacement original, EndPoint target) {
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
     * Returns a set of endpoints from the present data placement. These endpoints all overlap with endpoints in the newDataplacement.
     * @param present
     * @param newPlacement
     * @return
     */
    public static HashSet<EndPoint> getOverlap(DataPlacement present, DataPlacement newPlacement) {
        HashSet<EndPoint> result = new HashSet<>();
        if (present.getDataElement() == newPlacement.getDataElement()) {
            for (EndPoint target : newPlacement.getPlacement()) {
                for (EndPoint original : present.getPlacement()) {
                    if (doOverlap(original, target) && !sameGPU(original, target)) {
                        result.add(target);
                        result.add(original);
                    }
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
    private static boolean doOverlap(EndPoint first, EndPoint second) {
        if (first.getStart() <= second.getStart() && first.getStart() + first.getLength() >= second.getStart()) {
            return true;
        } else if (second.getStart() <= first.getStart() && second.getStart() + second.getLength() >= first.getStart()){
            return true;
        } else {
            return false;
        }
    }

    /**
     * Returns true, iff both endpoints run on the same GPU.
     * @param first
     * @param second
     * @return
     */
    private static boolean sameGPU(EndPoint first, EndPoint second) {
        if (first.getLocation() == second.getLocation() && first.getLocation().getType().equals("GPU")) {
            return true;
        }
        return false;
    }

    /**
     * Combines the original data placement with the data placement which is being read and returns it.
     * @param original
     * @param read
     * @return
     */
    public static DataPlacement createReadPlacements(DataPlacement original, DataPlacement read) {

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
    public static DataPlacement createWritePlacement(DataPlacement original, DataPlacement write) {
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
     * Returns the data identifier, iff the data is already present on the target GPU
     * @param necessaryData
     * @return
     */
    public static Optional<OffloadDataEncoding> isPresentOnGPU(DataPlacement necessaryData, HashSet<OffloadDataEncoding> onGPU) {
        for (OffloadDataEncoding GPUData: onGPU ) {
            if (isCovered(GPUData.getDataPlacement(), necessaryData)) {
                return Optional.of(GPUData);
            }
        }
        return Optional.empty();
    }

    /**
     * Returns true, iff onGPU fully covers the necessary Data
     * @param onGPU
     * @param necessaryData
     * @return
     */
    private static boolean isCovered(DataPlacement onGPU, DataPlacement necessaryData) {
        boolean DataPlacementCovered = true;
        if (onGPU.getDataElement() != necessaryData.getDataElement()) {
            return false;
        }
        for (EndPoint necessary: necessaryData.getPlacement() ) {
            boolean EndPointCovered = false;
            for (EndPoint present: onGPU.getPlacement() ) {
                if (present.getLocation() == necessary.getLocation()) {
                    EndPointCovered |= biggerOrEqual(present, necessary);
                }
            }
            DataPlacementCovered &= EndPointCovered;
        }
        return DataPlacementCovered;
    }

    /**
     * Returns true, iff present is larger than testing and covers the same chunk.
     * @param present
     * @param testing
     * @return
     */
    private static boolean biggerOrEqual(EndPoint present, EndPoint testing) {

        long startPresent = present.getStart();
        long endPresent = present.getStart() + present.getLength();

        long start = testing.getStart();
        long end = testing.getStart() + testing.getLength();

        if (start >= startPresent && end <= endPresent) {
            return true;
        }
        return false;
    }

    /**
     * combines a fractured data placement (a data placement with more than one endpoint).
     * The placement then contains the full span of the data placement.
     * @param fractured
     * @return
     */
    public static DataPlacement combineFracturedDataPlacements(DataPlacement fractured) {
        if (fractured.getPlacement().size() < 1) {
            Log.error("Placement to small");
            System.exit(1);
        }
        long start = findFirstData(fractured);
        long length = findLastData(fractured) - start;
        EndPoint endPoint = new EndPoint(fractured.getPlacement().get(0).getLocation(),start,length,new HashSet<>(),false);
        ArrayList<EndPoint> endPoints = new ArrayList<>();
        endPoints.add(endPoint);
        return new DataPlacement(endPoints, fractured.getDataElement());
    }

    private static long findFirstData(DataPlacement fractured) {
        long min = Long.MAX_VALUE;

        for (EndPoint point: fractured.getPlacement() ) {
            if (min > point.getStart()) {
                min = point.getStart();
            }
        }
        return min;
    }

    private static long findLastData(DataPlacement fractured) {
        long max = Long.MIN_VALUE;

        for (EndPoint point: fractured.getPlacement() ) {
            if (max < point.getStart() + point.getLength()) {
                max = point.getStart() + point.getLength();
            }
        }
        return max;
    }

    /**
     * Returns all overlapping encodings, iff the GPU data overlaps with the necessary data.
     * @param GPUData
     * @param necessaryData
     * @return
     */
    public static HashSet<OffloadDataEncoding> hasGPUOverlap(HashSet<OffloadDataEncoding> GPUData, HashSet<DataPlacement> necessaryData) {
        HashSet<OffloadDataEncoding> result = new HashSet<>();
        for (OffloadDataEncoding encoding: GPUData ) {
            for (DataPlacement placement: necessaryData ) {
                if (!getOverlap(encoding.getDataPlacement(), placement ).isEmpty()) {
                    result.add(encoding);
                    break;
                }
            }
        }
        return result;
    }
}
