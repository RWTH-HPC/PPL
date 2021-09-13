package de.parallelpatterndsl.patterndsl.dataSplits;

import com.google.common.collect.Lists;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The DataSplitTable traverses the original APT's methods to collect every data item.
 * The data items are then split according to the hyperparemeter and the splits are
 * stored in the table.
 * In particular, there must not be two data splits for the same indices of a single data item.
 */
public class DataSplitTable implements ExtendedShapeAPTVisitor {

    private DataSplitTable() {}

    private static DataSplitTable table = new DataSplitTable();

    private static Map<Data, ArrayList<DataSplit>> networkPackageTable = new HashMap<>();

    private static int dataSplitSize = 0;

    /**
     * Creates the DataSplitTable for the provided AbstractPatternTree.
     * @param apt - AbstractPatternTree to be analyzed.
     * @param dataSplitSize - split size.
     */
    public static void create(AbstractPatternTree apt, int dataSplitSize) {
        networkPackageTable.clear();
        DataSplitTable.dataSplitSize = dataSplitSize;

        for (Data data : apt.getGlobalVariableTable().values()) {
            if (data instanceof PrimitiveData) {
                create((PrimitiveData) data);
            } else if (data instanceof ArrayData) {
                create((ArrayData) data, dataSplitSize);
            }
        }
        apt.getRoot().accept(table.getRealThis());
    }

    /**
     * Creates a TempDataSplit object of given number of bytes.
     * @param bytes
     * @return TempDataSplit
     */
    public static TempDataSplit create(long bytes) {
        return new TempDataSplit(bytes);
    }

    private static void create(PrimitiveData data) {
        if (networkPackageTable.containsKey(data)) { return; }
        PrimitiveDataSplit pkg = new PrimitiveDataSplit(data);
        networkPackageTable.put(data, Lists.newArrayList(pkg));
    }

    private static void create(ArrayData data, int dataSplitSize) {
        if (networkPackageTable.containsKey(data)) { return; }
        ArrayList<DataSplit> pkgs = new ArrayList<>();
        int dataLength = data.getShape().get(0);
        for (int i = 0; i < dataLength; i += dataSplitSize) {
            int length = Integer.min(dataSplitSize, dataLength - i);
            ArrayDataSplit pkg = new ArrayDataSplit(data, i, length);
            pkgs.add(pkg);
        }
        networkPackageTable.put(data, pkgs);
    }

    public static Set<DataSplit> get(ArrayData data, int startIndex, int length) {
        if (!networkPackageTable.containsKey(data)) {
            return null;
        }

        ArrayList<DataSplit> pkgs = networkPackageTable.get(data);
        int startPackage = startIndex / dataSplitSize;
        int n = (int) Math.ceil((double) length / (double) dataSplitSize);
        return new HashSet<>(pkgs.subList(startPackage, startPackage + n));
    }

    public static DataSplit get(ArrayData data, int index) {
        if (!networkPackageTable.containsKey(data)) {
            return null;
        }

        int i = index / dataSplitSize;
        return networkPackageTable.get(data).get(i);
    }

    public static DataSplit get(PrimitiveData data) {
        if (!networkPackageTable.containsKey(data)) {
            return null;
        }
        return networkPackageTable.get(data).get(0);
    }

    @Override
    public void endVisit(SerialNode node) {
        Set<Data> ds = node.getChildren()
                .stream()
                .flatMap(p -> Stream.concat(p.getInputElements().stream(), p.getOutputElements().stream()))
                .collect(Collectors.toSet());

        for (Data data : ds) {
            if (data instanceof PrimitiveData) {
                create((PrimitiveData) data);
            } else if (data instanceof ArrayData) {
                create((ArrayData) data, dataSplitSize);
            }
        }
    }

    @Override
    public void endVisit(CallNode node) {
        Set<Data> ds = node.getChildren()
                .stream()
                .flatMap(p -> Stream.concat(p.getInputElements().stream(), p.getOutputElements().stream()))
                .collect(Collectors.toSet());

        for (Data data : ds) {
            if (data instanceof PrimitiveData) {
                create((PrimitiveData) data);
            } else if (data instanceof ArrayData) {
                create((ArrayData) data, dataSplitSize);
            }
        }
    }

}
