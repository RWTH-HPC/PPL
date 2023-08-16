package de.parallelpatterndsl.patterndsl;

import de.parallelpatterndsl.patterndsl.MappingTree.AbstractMappingTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.dataSplits.DataSplit;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Network;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Processor;
import de.parallelpatterndsl.patterndsl.patternSplits.PatternSplit;
import org.javatuples.Pair;
import org.javatuples.Tuple;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class GreedyMapping {

    Network cluster;
    FlatAPT APT;

    ArrayList<Pair<PatternSplit, Processor>> mapping = new ArrayList<>();

    HashMap<DataSplit, Device> variableMapping = new HashMap<>();

    public AbstractMappingTree generateMapping() {

        double currentCost = 0;
        for (int i = 0; i < APT.size() ; i++) {
            Set<PatternSplit> currentSet = APT.getSplits(i);

        }

        return null;
    }


    private double getTransferCost(DataSplit split, Device source, Device target) {
        long size = split.getBytes();
        double latency = 0;
        double bandwidth = Double.POSITIVE_INFINITY;
        latency += source.getParent().getConnectivityLatency(source.getParent().getDevices().get(0).getIdentifier(), source.getIdentifier());
        latency += target.getParent().getConnectivityLatency(target.getParent().getDevices().get(0).getIdentifier(), target.getIdentifier());

        bandwidth = Math.min(source.getParent().getConnectivityBandwidth(source.getParent().getDevices().get(0).getIdentifier(), source.getIdentifier()), bandwidth);
        bandwidth = Math.min(target.getParent().getConnectivityBandwidth(target.getParent().getDevices().get(0).getIdentifier(), target.getIdentifier()), bandwidth);

        latency += cluster.getConnectivityLatency(source.getIdentifier(), target.getIdentifier());
        bandwidth = Math.min(bandwidth, cluster.getConnectivityBandwidth(source.getIdentifier(), target.getIdentifier()));

        double duration = size/bandwidth + latency/1e9;
        return duration;
    }


    private double getExecutionCost(Processor exUnit, PatternSplit task) {
        double cost = task.getNode().getCost();
        double loadStore = task.getNode().getLoadStore();

        PatternNode node = task.getNode();
        if (node instanceof ParallelCallNode) {

            long taskSize = 0;
            for (int i = 0; i < task.getLengths().length; i++) {
                taskSize += task.getLengths()[i];
            }
            double fraction = taskSize / ((ParallelCallNode) node).totalIterations();
            cost *= fraction;
            loadStore *= fraction;
        }

        double flops = exUnit.getCores() * exUnit.getFrequency() * 1e9 * 2 * 8;
        double bandwidth = exUnit.getParent().getMainBandwidth();

        if (canUseCache(exUnit, task)) {
            bandwidth = exUnit.getCacheBandwidth().get(0);
        }

        double Performance = Math.min(cost/flops, loadStore/bandwidth);

        return Performance;

    }

    private boolean canUseCache(Processor exUnit, PatternSplit task){
        for (int i = mapping.size() - 1; i >= 0 ; i--) {
            Pair<PatternSplit, Processor> last = mapping.get(i);
            if (last.getValue1() == exUnit) {
                if (isCovered(last.getValue0().getInputDataSplits(), last.getValue0().getOutputDataSplits(), task.getInputDataSplits())) {
                    return true;
                } else {
                    return false;
                }
            }
        }
        return false;
    }

    private boolean isCovered(Set<DataSplit> lastIn, Set<DataSplit> lastOut, Set<DataSplit> current) {
        for (DataSplit split: current ) {
            if (!lastIn.contains(split) && !lastOut.contains(split)) {
                return false;
            }
        }
        return true;
    }
}
