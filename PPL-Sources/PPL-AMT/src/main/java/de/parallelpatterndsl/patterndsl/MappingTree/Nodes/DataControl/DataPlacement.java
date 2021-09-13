package de.parallelpatterndsl.patterndsl.MappingTree.Nodes.DataControl;

import de.parallelpatterndsl.patterndsl.MappingTree.DataMovementGenerator.ParallelGroup;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;

import java.util.ArrayList;
import java.util.HashSet;

/**
 * Class defining the placements of a specific data element.
 */
public class DataPlacement {


    private ArrayList<EndPoint> placement;

    private Data dataElement;

    public DataPlacement(ArrayList<EndPoint> placement, Data dataElement) {
        this.placement = placement;
        this.dataElement = dataElement;

        sortAndCompact();
    }

    public ArrayList<EndPoint> getPlacement() {
        return placement;
    }

    public Data getDataElement() {
        return dataElement;
    }

    /**
     * Sorts the array placement based on their start index and combines overlapping endpoints.
     */
    private void sortAndCompact() {
        ArrayList<EndPoint> result = new ArrayList<>();
        ArrayList<EndPoint> copy = new ArrayList<>(placement);

        while(!copy.isEmpty()) {
            EndPoint first = getSmallestElement(copy);
            copy.remove(first);
            ArrayList<EndPoint> sharedDevices = getSharedDevice(copy, first.getLocation());

            ArrayList<EndPoint> sharedResults = getSharedDevice(result, first.getLocation());

            if (!isCovered(first, sharedResults)) {
                first = maximizeElement(first, sharedDevices);
                result.add(first);
            } else {
                copy.remove(first);
            }
        }

        placement = result;
    }

    /**
     * Returns true, iff all elements from the endPoint are covered.
     * @param endPoint
     * @param array
     * @return
     */
    private boolean isCovered(EndPoint endPoint, ArrayList<EndPoint> array) {
        EndPoint iterator = endPoint;
        for (EndPoint element: array) {
            if (element.getStart() <= iterator.getStart() && element.getStart() + element.getLength() >= iterator.getStart()) {
                if (element.getStart() + element.getLength() >= iterator.getStart() + iterator.getLength()) {
                    return true;
                } else {
                    long start = element.getStart() + element.getLength();
                    long length = iterator.getStart() + iterator.getLength() - start;
                    iterator = new EndPoint(iterator.getLocation(),start,length,iterator.getParallelAccess(),iterator.isHasParallelWriteAccess());
                }
            }
        }
        return false;
    }

    /**
     * Returns the endpoint with the smallest start index from a list of endPoints.
     * @param array
     * @return
     */
    private EndPoint getSmallestElement(ArrayList<EndPoint> array) {
        EndPoint current = array.get(0);

        for ( EndPoint endpoint: array ) {
            if (current.getStart() > endpoint.getStart()) {
                current = endpoint;
            }
        }
        return current;
    }

    /**
     * Filters the list "endPoints" for all elements on the device "device".
     * @param endPoints
     * @param device
     * @return
     */
    private ArrayList<EndPoint> getSharedDevice(ArrayList<EndPoint> endPoints, Device device) {
        ArrayList<EndPoint> result = new ArrayList<>();
        for (EndPoint element: endPoints ) {
            if (element.getLocation() == device) {
                result.add(element);
            }
        }
        return result;
    }

    /**
     * Generates the maximum concatenated Endpoint, starting with start, based on the list shared. The elements in shared must be located on the same device.
     * @param start
     * @param shared
     * @return
     */
    private EndPoint maximizeElement(EndPoint start, ArrayList<EndPoint> shared) {
        ArrayList<EndPoint> partial = shared;
        EndPoint result = start;
            for (int i = 0; i < partial.size(); i++) {
                if (shared.get(i).getStart() >= result.getStart() && shared.get(i).getStart() <= result.getStart() + result.getLength()) {
                    HashSet<ParallelGroup> accesses = new HashSet<>(result.getParallelAccess());
                    accesses.addAll(shared.get(i).getParallelAccess());
                    result = new EndPoint(start.getLocation(), result.getStart(), getCombinedLength(result, shared.get(i)), accesses, result.isHasParallelWriteAccess() || shared.get(i).isHasParallelWriteAccess());
                }
            }

        return result;
    }

    /**
     * Returns the length of the resulting endPoint if first and second were to be combined.
     * @param first
     * @param second
     * @return
     */
    private long getCombinedLength(EndPoint first, EndPoint second) {
        long firstEnd = first.getStart() + first.getLength();
        long secondEnd = second.getStart() + second.getLength();

        long combinedEnd;

        if (firstEnd > secondEnd) {
            combinedEnd = firstEnd;
        } else {
            combinedEnd = secondEnd;
        }

        return combinedEnd - first.getStart();
    }

}
